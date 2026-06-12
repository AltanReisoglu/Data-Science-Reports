Overview
Datathon 2026 başlıyor! Kurallar, tanım, değerlendirme ve kod kısımlarını detaylı olarak incelemeyi unutmayın!

Datathon 2026 Ana Görev Dataset içerisinde öğrencilerin akademik, teknik, proje, portfolyo, mülakat ve kariyer hazırlık süreçlerine ait veriler bulunmaktadır. Veri setinde öğrencilerin hedef kariyer rolleri, teknik beceri skorları, proje deneyimleri, staj bilgileri, portfolyo ve GitHub aktiviteleri, iletişim ve takım çalışması gibi sosyal becerileri yer almaktadır.

Train veri setinde her öğrenci için career_success_score değeri verilmiştir. Bu skor, öğrencinin kariyer başarısını temsil eden 0-100 aralığında sürekli bir hedef değişkendir. Yarışmacılardan beklenen, test veri setindeki öğrenci profillerine göre her öğrencinin career_success_score değerini tahmin etmeleridir.

Veri setinde sayısal, kategorik ve doğal dil tabanlı alanlar birlikte bulunmaktadır. mentor_feedback_text alanı, öğrencinin gelişimi ve potansiyeli hakkında mentor perspektifinden yazılmış kısa bir değerlendirme metnidir. Bu nedenle yarışmacıların yalnızca klasik sayısal değişkenleri değil, doğal dil alanından gelen bilgiyi de modelleme sürecine dahil etmeleri beklenmektedir.

Dosyalar
train.csv - Training veri seti. Öğrenci bilgileri ve hedef değişken olan career_success_score alanını içerir.
test.csv - Test veri seti. Tahmin edilmesi gereken öğrenci bilgilerini içerir. Bu dosyada career_success_score alanı bulunmaz.
sample_submission.csv - Submit edilmesi gereken örnek dosya formatıdır.
Data ile ilgili daha fazla bilgi için Data sekmesini inceleyebilirsiniz.

Genel Akış:

Aşama: Kaggle online yarışma 9 Haziran 20.00 - 14 Haziran 23.59
Aşama: Kaggle'da ilk 10'a girmeye hak kazanan takım ve bireylerin online olarak jüriye sunum yapması (tarihleri ve formatı finale kalan takımlara ayrıca belirtilecektir). Yarışmanın ödüllerini kazanacak ilk 3 takım, sunum sonrası değerlendirmeden sonra belirlenecektir.
Aşama: İlk 3'e giren takım / bireylerin BTK Akademi bünyesinde düzenlenecek olan ödül töreni ve Veri Bilimi Zirvesi'ne davet edilmesi (tarihleri ilgili takımlara iletilecektir)
Start

3 days ago
Close

2 days to go
Description
Google ve Girişimcilik Vakfı ile birlikte düzenlediğimiz Türkiye'de veri bilimine gönül vermiş herkese açık ve öğrendiklerini pratiğe dökme şansını yakalayacağı veri yarışmamız Datathon 2026 başlıyor!

BTK Akademi çevrim içi eğitim portalı üzerinden tüm kayıtlı öğrencilere istatistik, veri bilimi, veri analizi, büyük veri, makine öğrenmesi, derin öğrenme, Python programlama gibi konularda çeşitli eğitimler uzun bir süredir sunulmaktadır. Bu eğitimleri alan ve veri bilimi alanına giriş yapmak isteyen öğrenciler için Google ve Girvak iş birliğiyle 2025 Datathon planlaması yapılmıştır. İlki 2022 yılında gerçekleştirilen Datathon aktivitemiz 2026 yılında da tekrar Türkiye'de veri bilimine gönül vermiş herkese açık olarak yayınlanmıştır.

Datathon'un amacı veri bilimine giriş yapmakta olan öğrencilerin teorik bilgilerini pratiğe dökmesi için bir fırsat sunmanın yanı sıra endüstri profesyonellerinin de kendi bilgilerini test etmesine olanak tanımaktır. BTK Akademi içerisinde Datathon aktivitemizi ve aynı zamanda veri bilimi ile ilgili diğer tüm bilgileri destekleyecek içerikler hali hazırda ücretsiz olarak kullanıma sunulmuştur.

Yarışmaya katılmadan önce kurallar bölümünün mutlaka incelenmesini tavsiye ederiz.

Yarışmada resmi olarak ilk 10'a kalıp jüriye sunum yapabilme hakkını elde etmek için BTK Akademi üzerinden başvurunuzu yapmış olmanız gerekmektedir. Başvuru yapmayan kişiler veya takımlar elenecek ve ilk 10 ekip BTK Akademi tarafından yarışma bittikten sonra ayrıca duyurulacaktır.

Not: İlk 10 takımın belirlenmesi için BTK Akademi yarışmacı takımlardan çalışma dosyalarının haricinde ek materyal talep edebilir.

Evaluation
Bu yarışma, her bir örnek verinin career_success_score değerini doğru bir şekilde tahmin etmeyi amaçlamaktadır. Yarışmanın başarısını ölçmek için kullanılacak metrik ise MSE olacaktır.

Ortalama kare sapma (MSE), bir yanda gerçek ya da tahmin edilen değerler ile diğer yanda gözlemlenen değerler ya da bir tahmin edici arasındaki farkların yakından ilişkili ve sıklıkla kullanılan ölçümlerden biridir.