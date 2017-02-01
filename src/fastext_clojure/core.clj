(ns fastext-clojure.core
  (:use [uncomplicate.neanderthal core native]
        [clojure.math.numeric-tower])
  (:gen-class))

(def max-sigmoid 8)

(def sigmoid-table-size 512)

(def log-table-size 512)

(defn make-ngram-hash
  [sequence-of-chars word-id]
  (let [string (apply str sequence-of-chars)]
    (+ word-id (/ (hash [string]) 2000000))))

(defn get-subwords
  [word word-id]
  (let [processed-word (str "<" word ">")]
    (->> (map #(partition % 1 processed-word) (range 3 7))
         (apply interleave)
         (map #(make-ngram-hash % word-id)))))

(defn ->vocabulary
  [words]
  (reduce
   (fn [acc word]
     (let [hash-of-word (hash [word])
           word-id      (count acc)
           base-map     {:word      word
                         :word-hash hash-of-word
                         :count     1
                         :type      0
                         :subwords  (into [word-id] (get-subwords word word-id))}]

       (update-in acc [hash-of-word] (fn [old-map]
                                       (assoc base-map :count (inc (or (:count old-map) 0)))))))
   {}
   words))

(defn ->unigram-table
  [counts]
  (let [weighted-counts     (into {} (map-indexed (fn [idx count] [idx (expt count 0.5)]) counts))
        sum-weighted-counts (apply + (vals weighted-counts))
        table-size          10000000
        unigram-table       (reduce
                             (fn [unigram-table [index count]]
                               (into unigram-table (repeat (/ (* count table-size) sum-weighted-counts) index)))
                             []
                             weighted-counts)]
    (shuffle unigram-table)))

(defn ->context-window
  [token-count total-token-count]
  (let [upper-bound    (+ 1 (rand-int 5))
        lower-bound    (* -1 upper-bound)
        context-window (->> (range (+ token-count lower-bound) (inc (+ token-count upper-bound)))
                            (remove zero?)
                            (remove neg?)
                            (remove #(> % total-token-count)))]
    context-window))

(defn compute-output-matrix
  [output-matrix hidden-layer target alpha]
  (doseq [index (range 100)]
    (do
      (alter! output-matrix target index (fn ^double [^double x]
                                           (double (+ x (* alpha (entry hidden-layer index)))))))))

(defn compute-gradient
  [output-matrix gradient target alpha]
  (doseq [index (range 100)]
    (do
      (alter! gradient index (fn ^double [^double x]
                               (double (+ x (* alpha (entry output-matrix target index)))))))))

(defn compute-hidden
  [hidden-layer input-matrix ngrams]
  (doseq [ngram ngrams]
    (doseq [ngram-index (range 100)]
      (do
        (alter! hidden-layer ngram-index (fn ^double [^double x]
                                           (double (*
                                                    (+ x (entry input-matrix ngram ngram-index))
                                                    (/ 1.0 (count ngrams))))))))))

(defn init-sigmoid
  [x]
  (let [y (/ (* x 2 max-sigmoid) (- sigmoid-table-size max-sigmoid))]
    (/ 1.0 (+ 1.0 (Math/exp (* -1.0 y))))))

(defn init-log
  [x]
  (let [y (/ (+ x 1e-5) log-table-size)]
    (Math/log y)))

(def sigmoid-table (dv (map #(init-sigmoid %) (range 0 513))))
(def log-table (dv (map #(init-log %) (range 0 513))))

(defn sigmoid
  [x]
  (cond
    (< x (* -1 max-sigmoid))
    0.0

    (> x max-sigmoid)
    1.0

    :else
    (let [sigmoid-index (/ (* (+ x max-sigmoid) sigmoid-table-size) max-sigmoid 2)]
      (entry sigmoid-table sigmoid-index))))

(defn get-log
  [x]
  (cond
    (> x 1.0)
    0.0

    :else
    (let [log-index (int (* x log-table-size))]
      (entry log-table log-index))))

(defn compute-loss
  [output-matrix gradient hidden-layer target label learning-rate]
  (let [score (sigmoid (dot (row output-matrix target) hidden-layer))
        alpha (* learning-rate (- label score))]
    (compute-gradient output-matrix gradient target alpha)
    (compute-output-matrix output-matrix hidden-layer target alpha)
    (println score)
    (case label
      1.0
      (* -1.0 (get-log score))

      (* -1.0 (get-log (- 1.0 score))))))


(defn get-negative
  [unigram-table target-token]
  (let [random-token (rand-nth unigram-table)]
    (if (= random-token target-token)
      (get-negative unigram-table target-token)
      random-token)))

(defn -main
  [& args]

  (with-open [rdr (clojure.java.io/reader (first args))]
    (let [file (line-seq rdr)]
      (let [string-file          (clojure.string/trim (first file))
            words                (-> (clojure.string/split string-file #" " 1000001)
                                     (drop-last))
            vocabulary           (->> (vals (->vocabulary words))
                                      (filter #(> (:count %) 5))
                                      (sort-by :count >))
            hash->id             (into {} (map-indexed (fn [idx word] [(:word-hash word) idx]) vocabulary))
            parsed-text          (remove nil? (map #(get hash->id (hash [%]) nil) words))
            nwords               (count vocabulary)
            ntokens              (count words)
            dimmension           100
            input-matrix         (dge (+ nwords 2000000) dimmension (repeatedly #(double (/ (- (rand) 0.5) 50))))
            output-matrix        (dge nwords dimmension)
            unigram-table        (->unigram-table (map :count vocabulary))
            discard-table        (map (fn [word]
                                        (let [frequency          (/ (:count word) ntokens)
                                              weighted-frequency (/ 0.0001 frequency)]
                                          (+ (sqrt weighted-frequency) weighted-frequency))) vocabulary)
            filtered-parsed-text (filter (fn [token] (> (rand) (nth discard-table token))) parsed-text)]



        (doall (map-indexed (fn [token-count token]
                              (let [total-num-tokens (count filtered-parsed-text)
                                    progress         (/ token-count total-num-tokens)
                                    learning-rate    (* 0.025 (- 1.0 progress))
                                    ngrams           (:subwords (nth vocabulary token))
                                    context-window   (->context-window token-count total-num-tokens)]

                                (when-not (empty? ngrams)
                                  (mapv (fn [pt-index]


                                          (let [hidden-layer (dv dimmension)
                                                gradient     (dv dimmension)]

                                            ;; Compute hidden
                                            (compute-hidden hidden-layer input-matrix ngrams)

                                            ;; Compute negative sampling
                                            (let [pt-token (nth filtered-parsed-text pt-index)
                                                  loss     (reduce
                                                            (fn [loss index]
                                                              (case index
                                                                0
                                                                (+ loss (compute-loss output-matrix gradient hidden-layer pt-token 1.0 learning-rate))

                                                                (+ loss (compute-loss output-matrix gradient hidden-layer (get-negative unigram-table pt-token) 0.0 learning-rate))))
                                                            0.0
                                                            (range 0 6))]
                                              (println loss)))) context-window)))) filtered-parsed-text))

        (println (count parsed-text))
        (println (count filtered-parsed-text))))))
