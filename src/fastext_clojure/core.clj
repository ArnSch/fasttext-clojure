(ns fastext-clojure.core
  (:use [uncomplicate.neanderthal core native]
        [clojure.math.numeric-tower])
  (:gen-class))

(defn make-ngram-hash
  [sequence-of-chars]
  (let [string (apply str sequence-of-chars)]
    (hash [string])))

(defn get-subwords
  [word]
  (let [processed-word (str "<" word ">")]
    (->> (map #(partition % 1 processed-word) (range 3 7))
         (apply interleave)
         (map #(make-ngram-hash %)))))

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
                         :subwords  (into [word-id] (get-subwords word))}]

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
            input-matrix         (dge (+ nwords 2000000) 100 (repeatedly #(/ (- (rand) 0.5) 50)))
            output-matrix        (dge nwords 100)
            unigram-table        (->unigram-table (map :count vocabulary))
            discard-table        (map (fn [word]
                                        (let [frequency          (/ (:count word) ntokens)
                                              weighted-frequency (/ 0.0001 frequency)]
                                          (+ (sqrt weighted-frequency) weighted-frequency))) vocabulary)
            filtered-parsed-text (filter (fn [token] (> (rand) (nth discard-table token))) parsed-text)]


        ;;; TODO: REMEMBER TO TEST FOR NIL WHEN USING PARSED TEXT


        (println (count parsed-text))
        (println (count filtered-parsed-text))))))
