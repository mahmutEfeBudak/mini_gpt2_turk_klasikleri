# miniGPT ile Türk Klasikleri Üzerinde Dil Modeli Eğitimi

Bu proje, Andrei Karpathy'nin "Let's build GPT" video dersi ve [ng-video-lecture](https://github.com/karpathy/ng-video-lecture) reposu temel alınarak yazılmış bir GPT mimarisinin, Türk klasik edebiyatı metinleri üzerinde sıfırdan eğitilmesi denemesidir.

## Motivasyon

Türkçe için pretraining sürecini uçtan uca (veri toplamadan model eğitimine kadar) deneyimlemek amaçtı.

## Veri Toplama ve Temizleme

Proje de en çok zorlandığım yer burasıydı. Türk edebiyatı klasiklerini bulmak ve metinleri modelin öğrenmesi için olabildiğince OCR ve dil bilgisi hatalarından ayıklamak çok zahmetliydi. 

- **Karşılaşılan sorunlar:**
 İnternet üzerinde klasik metin bulmak çok kısıtlıydı. Bunun yanında pdf' leri 'txt' dosyasına çevirmenin getirdiği başka sorunlarda vardı. Bunlar OCR ve dilbilgisi hatalarıydı. Sayfa numaraları ve başlıkar birbirlerine çok karışmıştı. Verinin içine dipnotlarda girmişti. Model kelimelerin yanında bulunan sayfa numaralarını öğrendiği için çıktıda alakasız numaralar bulunuyordu. Örnek: "Ahmet Cemil bir gün145 yolda giderken Rauf ile98 karşılaşmıştı.295". Yaklaşık 20 mb' lık bir dosyada bu verileri manuel temizlemek gerçekçi olmadığı için AI' dan yardım aldım. AI kullanmanın da getirdiği zorluklar vardı: Veri OCR hatalarından ayıklanırken başka OCR hatalarının eklenmesi, bazı türkçe karakterlerin türkçeden sayılmaması, dipnotlardan temizlenirken önemli verilerin bazılarının silinmesi, konuşma metinlerinin tek bir paragraf haline getirilmesi vb.

- **Çözüm yaklaşımı:**
  - Dosyaları hem tek başlarına hem de düzeltilmiş versiyonlarıyla karşılaştırıp analiz ettirdim. Çıkan analize göre AI' dan dosyayı tekrardan oluşturmasını istedim.

- **Çözemediğim / tam çözülmemiş sorunlar:**
  - Hâlâ OCR hataları bulunuyor("yo ld a gider k en karşılaştığı kişiyle kavgaya125..."). Ayrıca konuşma metinlerinin birçoğu tek bir paragraf olarak geçiyor. 

## Model ve Eğitim Detayları

| Parametre | Değer |
|---|---|
| Model boyutu | 6 layer, 6 head, 384 embd |
| Context length | 1024 |
| Eğitim adımı (steps) | 60000 |
| Donanım | Kaggle T4X2 GPU'ları |
| Eğitim süresi | ~8 saat |

## Sonuçlar

Modelin ürettiği örnek çıktı:


>Kadın cevap vermedi” diye korkuyla sordu, “Seni müdürüne haber vereceğini söyleyeyim: Sözlerimde, -Sen bu soru üzerine neler verdin?” dedi. Maamafih, biraz düşündükten sonra, “Biraz evvel gidiyorum” dedi “Haydi Çalıkuşu, bu haydi!”

>Bu sözü dışarı at. Ne söylediğimi anlamıştım. Masanın üstüne oturdum. -Dilsitan Müdürü muayenehanesi mi? -Benim sözümü söyleme bu kadının etrafında her bir kız gibi Ferhunde söylüyor, bahçeye koşuyor. Ben, güzel dolu bir karış karış dönüyorum, fakat bahçede kimsenin nesi, bizim “otur” diye de evvelce görmek mümkün olmayan bir yer olmalı. Dışarda bir karış ki, bitkin yer taşıman, böyle küçük dalların ululuğudur, bütün muharrir ve dağlar Ayasofya’yı en az az süzmeye alıştım. Bunları mahkûm ettiler. Fakat kim ölmüş? Ha, doğru terbiyeli, mahzuniyeti idi. Hüsrev alim, kimseleri de yoktu. O gün onu ona söylemekte belki artık mazinin, bu ağaç, bu tür duvarın etrafındaki ağaçta oturma büyümüştü. Sessizce duruyor, doktorlar, Sör Hemşin’in dümende mütenasip bir adam


Model türkçe gramer yapısını kısmen öğrenmiş gözüküyor ama cümle ve paragrafın kendi içinde anlam bütünlüğünü oluşturamıyor. Bunun sebepleri:

- Veri setinin oldukça küçük olması. Hali hazırda türkçenin kendisi sondan eklemeli bir dil olduğu için ingilizce gibi dillerden daha zorken Türk edebiyatı klasiklerinin dili soyut betimlemelerle zengin olduğundan modelin tokenlar arası ilişki kurması çok daha zor hâle geldi. Bunun için çok büyük datasetleri gerekiyordu. 

- Veri setinin küçük olmasından dolayı tokenizer karakter bazlı seçildi. BPE tokenizerlı model bu datasette çalıştığında tokenlar arası anlam ilişkilerini kurmakta zorlandı. Ortaya çıkan sonuç temel gramer yapısını bile öğrenememişti. Karakter bazlı olunca kelimler ve ekleri oluşturmada daha başarılı oldu. 

- Modelin parametre sayısı mevcut dataset için yeterliydi. Daha fazla parametre sadece overfitting'e sebep olurdu. Veri seti azlığı ve modelin karakter tokenizer bazlı olması modelin öğrenmesini kısıtlayan asıl unsurlardı. Loss değerlerinin düşüşleri tek başlarına çıktı kalitesi için asıl unsur olmaması da sebeplerden bir tanesi.

## Ne Öğrendim

- Veri kalitesi ve çokluğu model eğitimi için asıl unsurdu. Ne kadar çok veriye sahip olunursa o kadar karmaşık bir dil modeli oluşturulabilir ve tokenlar arası anlam çok daha zenginleşebilirdi.

- Büyük datasetleri ile uğraşırken AI ve regex kullanımı esansiyel hâle geldi.

- En önemlisi Loss değerlerinin tek başlarına çıktı kalitesi için kriter olmadığıydı. Karakter bazlı tokenizer'a sahip olmanın getirdiği dezavantajlardan da birisiydi bu. Model kelimeleri ve birden fazla kelimeden oluşan grupları düzgün öğrenmişken çokça saçmalıyordu. Context uzunluğunu yeterince uzatırsam bu problemden kurtulabileceğimi düşündüm ama veri setinin az olması ve modelin bir sonraki karakteri öğrenme bazlı olmasından dolayı problem çözülemedi.

## Kaynaklar

- Model mimarisi: [Andrej Karpathy - Let's build GPT](https://github.com/karpathy/ng-video-lecture)

**Not** -: Orijinal repo açık bir lisans içermemektedir. Model mimarisi referans alınarak öğrenme amacıyla yazılmıştır...