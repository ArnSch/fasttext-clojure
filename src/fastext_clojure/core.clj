(ns fastext-clojure.core
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :refer :all])
  (:gen-class))

(set-current-implementation :vectorz)

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

(defn -main
  [& args]

  (with-open [rdr (clojure.java.io/reader (first args))]
    (let [file (line-seq rdr)]
      (let [string-file (clojure.string/trim (first file))
            words       (-> (clojure.string/split string-file #" " 1000001)
                            (drop-last))
            vocabulary  (->> (vals (->vocabulary words))
                             (filter #(> (:count %) 5))
                             (sort-by :count >))
            hash->id    (into {} (map-indexed (fn [idx word] [(:word-hash word) idx]) vocabulary))
            parsed-text (map #(get hash->id (hash [%]) nil) words)]

        ;;; TODO: REMEMBER TO TEST FOR NIL WHEN USING PARSED TEXT

        (println (nth vocabulary 173))))))
