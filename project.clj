(defproject fastext-clojure "0.1.0-SNAPSHOT"
  :jvm-opts ["-Xmx6g"]

  :dependencies [[org.clojure/clojure "1.8.0"]
                 [uncomplicate/neanderthal "0.8.0"]
                 [net.mikera/core.matrix "0.57.0"]
                 [org.clojure/math.numeric-tower "0.0.4"]]
  :main ^:skip-aot fastext-clojure.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
