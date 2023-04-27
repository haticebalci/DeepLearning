# DeepLearning
DeepLearning
Bu, bir sınıflandırma problemi için Keras kütüphanesini kullanarak derin öğrenme modeli oluşturmak için Python'da yazılmış bir koddur. Iris veri kümesi kullanılır ve kod, veri ön işleme adımlarını içerir, örneğin veri kümesinin eğitim ve test setlerine bölünmesi, kategorik bağımlı değişkenin tekli kodlanması ve bağımsız değişkenlerin standartlaştırılması.

Kod, Pandas, NumPy, Matplotlib ve Keras gibi gerekli kütüphaneleri içe aktarır. Veri kümesi, Pandas'ın read_excel yöntemi kullanılarak içe aktarılır. Bağımlı ve bağımsız değişkenler, sırasıyla bir matris ve vektöre ayrılır. Bağımlı değişken, kategorik olduğundan tekli kodlama uygulanır. Veri kümesi, sklearn.model_selection kütüphanesinden train_test_split yöntemi kullanılarak eğitim ve test setlerine bölünür.

Bağımsız değişkenler, sklearn.preprocessing kütüphanesinden StandardScaler yöntemi kullanılarak standartlaştırılır. Keras kütüphanesinin Sequential modeli kullanılarak sıralı bir model oluşturulur. Girdi ve gizli katmanlar, add yöntemi kullanılarak modele eklenir ve çıktı katmanı sonradan eklenir. Katmanlar, nöron sayısı, aktivasyon fonksiyonları ve çekirdek başlatıcısı gibi belirli parametrelerle tanımlanır. Gizli katmanda relu fonksiyonu kullanılır ve çıktı katmanında softmax fonksiyonu kullanılır.

Model oluşturulduktan sonra, optimize edilir. compile yöntemi kullanılarak üç ana parametre: optimize edici, kayıp fonksiyonu ve metrikler belirtilir. Adam optimize edici fonksiyonu kullanılır, kategorik çapraz entropi kayıp fonksiyonu kullanılır ve doğruluk metriği kullanılır.

Son olarak, model fit yöntemi kullanılarak eğitilir ve performansı test seti kullanılarak değerlendirilir.
