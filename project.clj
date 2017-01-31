(defproject fastext-clojure "0.1.0-SNAPSHOT"
  :jvm-opts ["-Xmx6g"]

  :dependencies [[org.clojure/clojure "1.8.0"]
                 [net.mikera/vectorz-clj "0.45.0"]]
  :main ^:skip-aot fastext-clojure.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
