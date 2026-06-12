
Dataset Description
Dataset içerisinde öğrencilerin akademik, teknik, proje, portfolyo, mülakat ve kariyer hazırlık süreçlerine ait veriler bulunmaktadır. Veri setinde öğrencilerin hedef kariyer rolleri, teknik beceri skorları, proje deneyimleri, staj bilgileri, portfolyo ve GitHub aktiviteleri, iletişim ve takım çalışması gibi sosyal becerileri yer almaktadır.

Train veri setinde her öğrenci için career_success_score değeri verilmiştir. Bu skor, öğrencinin kariyer başarısını temsil eden 0-100 aralığında sürekli bir hedef değişkendir. Yarışmacılardan beklenen, test veri setindeki öğrenci profillerine göre her öğrencinin career_success_score değerini tahmin etmeleridir.

Veri setinde sayısal, kategorik ve doğal dil tabanlı alanlar birlikte bulunmaktadır. mentor_feedback_text alanı, öğrencinin gelişimi ve potansiyeli hakkında mentor perspektifinden yazılmış kısa bir değerlendirme metnidir. Bu nedenle yarışmacıların yalnızca klasik sayısal değişkenleri değil, doğal dil alanından gelen bilgiyi de modelleme sürecine dahil etmeleri beklenmektedir.

Dosyalar
train.csv - Training veri seti. Öğrenci bilgileri ve hedef değişken olan career_success_score alanını içerir.
test.csv - Test veri seti. Tahmin edilmesi gereken öğrenci bilgilerini içerir. Bu dosyada career_success_score alanı bulunmaz.
sample_submission.csv - Submit edilmesi gereken örnek dosya formatıdır.
Columns
student_id - Öğrencinin benzersiz kimliği.
application_year - Öğrencinin başvuru veya değerlendirme yılı.
age - Öğrencinin yaşı.
graduation_year - Öğrencinin mezuniyet yılı.
department - Öğrencinin bölümü.
university_tier - Öğrencinin mezun olduğu üniversitenin seviye kategorisi.
cgpa - Öğrencinin genel not ortalaması.
english_exam_score - Öğrencinin İngilizce sınav skoru.
attendance_rate - Öğrencinin eğitim veya programa devam oranı.
failed_courses_count - Öğrencinin başarısız olduğu ders sayısı.
target_role - Öğrencinin hedeflediği kariyer rolü.
coding_score - Kodlama becerisi skoru.
problem_solving_score - Problem çözme becerisi skoru.
data_structures_score - Veri yapıları bilgisi skoru.
sql_score - SQL becerisi skoru.
machine_learning_score - Makine öğrenmesi becerisi skoru.
backend_score - Backend geliştirme becerisi skoru.
frontend_score - Frontend geliştirme becerisi skoru.
cloud_score - Bulut teknolojileri becerisi skoru.
devops_score - DevOps becerisi skoru.
project_quality_score - Öğrencinin proje kalitesini temsil eden skor.
real_client_project_count - Öğrencinin gerçek müşteri veya gerçek ihtiyaç üzerinden geliştirdiği proje sayısı.
internship_count - Öğrencinin yaptığı staj sayısı.
internship_duration_months - Öğrencinin toplam staj süresi.
freelance_project_count - Öğrencinin freelance olarak yaptığı proje sayısı.
hackathon_count - Öğrencinin katıldığı hackathon sayısı.
hackathon_awards - Öğrencinin hackathonlarda kazandığı ödül sayısı.
portfolio_score - Öğrencinin portfolyo kalitesini temsil eden skor.
github_repo_count - Öğrencinin GitHub repo sayısı.
github_avg_stars - Öğrencinin GitHub repolarının ortalama yıldız sayısı.
open_source_contribution_count - Öğrencinin açık kaynak katkı sayısı.
linkedin_profile_score - Öğrencinin LinkedIn profil kalitesini temsil eden skor.
cv_quality_score - Öğrencinin CV kalitesini temsil eden skor.
technical_interview_score - Öğrencinin teknik mülakat başarısını temsil eden skor.
hr_interview_score - Öğrencinin insan kaynakları mülakat başarısını temsil eden skor.
communication_score - Öğrencinin iletişim becerisi skoru.
teamwork_score - Öğrencinin takım çalışması becerisi skoru.
leadership_score - Öğrencinin liderlik becerisi skoru.
presentation_score - Öğrencinin sunum becerisi skoru.
certification_count - Öğrencinin sahip olduğu sertifika sayısı.
bootcamp_count - Öğrencinin katıldığı bootcamp sayısı.
applications_sent - Öğrencinin yaptığı iş başvurusu sayısı.
interviews_attended - Öğrencinin katıldığı mülakat sayısı.
hobby - Öğrencinin hobisi.
preferred_social_media_platform - Öğrencinin tercih ettiği sosyal medya platformu.
mentor_feedback_text - Öğrencinin gelişimi, teknik yaklaşımı, iletişimi ve kariyer potansiyeli hakkında mentor değerlendirmesi.
career_success_score - Öğrencinin kariyer başarısını temsil eden hedef değişken. Bu alan yalnızca train.csv dosyasında bulunur.