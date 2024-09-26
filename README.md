# Driving Simulator

## Instrukcije i objašnjenja

- Preuzeti i instalirati anacondu https://www.anaconda.com/
- pokrenuti anaconda prompt
- izvršiti komande:
```
conda create --name drivesim-env python=3.9
conda activate drivesim-env
```
Instalirati biblioteke:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python matplotlib scipy pillow h5py natsort tqdm
```
po potrebi:

`pip install wandb`


Skup podataka (Comma2k19) se sastoji iz 10 čankova, svaki od kojih je zip fajl unutar kojeg se nalazi folder sa istim imenom.
Za skup za obuku je potrebno kreirati 'data' folder u 'drive_reader' folderu i ekstraktovati prva četiri čanka u njega, tako da 'data' folder sadrži
foldere Chunk_1 ... Chunk_4
Za test skup je potrebno kreirati 'data_test' folder 'drive_reader' folderu i ekstraktovati peti čank u njega, tako da 'data_test' folder sadrži
foldere Chunk_5

### Podešavanje Pycharm okruženja:
- preuzeti i instalirati Pycharm https://www.jetbrains.com/pycharm/
- otvoriti projekat: File->Open->'driving-simulator putanja'
- podesiti okruženje: File->Settings-> Pretražiti 'python interpreter'
	- padajući meni->show all->drivesim-env->Ok


Zatim je potrebno pokrenuti `export_dataset.py` skriptu u drivesim-env okruženju koja će pritom generisati h5 fajl skupa za obuku
Nakon toga, potrebno je pokrenuti `export_test_set.py` skriptu koja če generisati h5 fajl test skupa

Ove dve skripte prolaze kroz sve čankove u ciljanom folderu, unutar kojih se nalaze folderi sa snimcima različitih ruta. Skripta dalje
prolazi kroz svaki od foldera ruta, unutar kojih se nalaze folderi sa segmentima snimka te rute u obliku `video.hvec` fajla. U istom folderu
se takođe nalaze putanje processed_log->CAN->speed i processed_log->CAN-steering_angle. Ovi folderi sadrže informacije o vremenskim trenucima
`t` fajl, i snimljene vrednosti u `value` fajlu. Skripta za svaki segment, svake rute, svakog čanka uzima vremenske trenutke i vrednosti brzine
i ugla volana, interpolira ih i uzrokuje u trenucima koji se poklapaju sa trenucima video fajla. Zatim prolazi kroz sve frejmove video fajla,
iseca gornju i donju trećinu frejma, i menja dimenzije na 544x136. Za kraj, eksportuje niz frejmova, brzine i uglove volana u navedeni h5
fajl.

### VQ-VAE obuka:
Potrebno je otvoriti `train_vqvae.py` skriptu u 'vq_vae' folderu. Unutar skripte se nalazi main funkcija, gde se može podesiti putanja do
skupa za obuku. Ovo se može postaviti menjanjem vrednosti promenljive 'dataset_path' na putanju do prethodno generisanog h5 fajla za obuku.
Nakon toga, dovoljno je pokrenuti ovu skriptu unutar Pycharm-a, koja će zatim započeti obuku VQ-VAE modela.

VQ-VAE model ima ulogu da vrši tokenizaciju skupa za obuku, kako bi bilo moguće obučiti transformer model. Skritpa sadrži i neke
eksperimentalne delove koda koji na kraju nisu korišteni tokom obuke, npr. diskriminator i sobel filter. Na njih se ne treba obazirati.
Prvo što skripta radi je postavljanje parametara VQ-VAE modela. Zatim, kreira se dataset promenljiva tipa klase ImageDataset definisane u
`image_dataset.py` fajlu. Zatim se, na osnovu nasumičnih 1000 uzoraka iz skupa za obuku procenjuje varijansa, koja će se kasnije koristiti
za normalizaciju podataka. Konačno, skripta pokreće glavnu petlju za obuku koja pomoću train_loader dataloader-a dobavlja beč slika,
i poziva model nad njima kako bi se dobio izlaz. Zatim, računa se rekonstrukcioni gubitak kao srednja kvadratna greška između originalne slike
i dobijenog izlaza i normalizuje se sa prethodno izračunatom procenom varijanse. Nakon toga, računa se ukupan gubitak kao zbir
rekonstrukcionog gubitka i commitment gubitka, pomnoženog sa faktorom beta. Commitment gubitak se računa unutar `vqvae.py` fajla i predstavlja
srednju kvadratnu grešku između izlaza iz enkodera i najbližeg kvantizovanog embeding vektora iz rečnika. Njegov zadatak je da osigura
da izlazi iz enkodera ostanu blizu embeding vektorima rečnika.

Zatim, poziva se vq_vae_loss.backward(), koji akumulira gradijent N koraka (koji je u ovom slučaju 1), i svakih N iteracija se poziva
optimizer.step() koji vrši optimizacioni korak i ažurira težine modela. Svakih save_every koraka se čuva model u obliku .pth fajla.
Nije potrebno čekati da prođe svih 1000 epoha, Posle par epoha, kada gubitak prestane značajno da se smanjuje se može prekinuti obuka i
koristiti poslednji sačuvani .pth fajl modela.

### MSG-GAN obuka:
Uloga MSG-GAN-a u ovom radu je da na osvnovu kvantizovanih embeding vektora VQ-VAE enkodera, rekonstruiše originalnu sliku, tako da
nedoumice nastale gubitkom informacija prilikom kvantizacije, rešava dodavanjem detalja koji možda nisu prisutni u originalnoj slici.
Za obuku MSG-GAN-a u vq_vae folderu se može pokrenuti skripta `msg_gan_upscaler2.py`. U njemu se takođe može promeniti vrednost promenljive
dataset_path na putanju generisanog h5 skupa za obuku. U main funkciji se takođe nalazi linija vqvae.load_state_dict(torch.load('vqvae_v4_2999.pth')),
koja učitava prethodno obučen VQ-VAE. Ukoliko je ime ovog .pth fajla drugačije, npr. zbog različitog broja koraka, ovde je potrebno izmeniti
tu putanju. MSG-GAN se sastoji iz generatora i 3 diskriminatora. Generator ima 3 izlaza za 3 različite rezolucije, dok diskriminatori kao
ulaz dobijaju sliku rezolucije kao jedan od izlaza generatora i vrše binarnu klasifikaciju kojom određuju da li je slika autentična ili ne.
Generator se sastoji iz konvolucionih i batch_norm slojeva i ne sadrži rezidualne konekcije jer se nakon svakog bloka nalazi po jedan izlaz,
pa gradijentu nije problem da teče do najranijih slojeva. S druge strane, diskriminator, pored konvolucionih i batch norm slojeva, sadrži i
rezidualne konekcije.

Na početku koda se vrši inicijalizacija parametara obuke, kreacija dataloadera, učitavanje prethodno obučenog VQ-VAE modela i postavka lambda
parametra na 0.001. Ovaj parametar služi da skalira gubitak od diskriminatora, i koristi se da kontroliše slobodu generatora da dodaje
detalje koji se razlikuju od originalne slike. U main funkciji, inicijalizuje se generator i 3 diskriminatora, kao i optimizatori. Zatim se
učitava prethodno obučen VQ-VAE model. Konačno, ulazi se u glavnu petlju za obuku koja dobavlja beč pomoću dataloader-a, normalizuje
vrednosti piksela slike, poziva enkoder i kvantizator VQ-VAE-a sa nenormalizovanim slikama, i zatim prosleđuje njegov izlaz u generator.
Time se dobijaju rekonstruisane slike. Za svaku rekonstruisanu sliku se za sve 3 rezolucije računa L_rec kao srednja kvadratna greška
između originala i rekonstrukcije, kao i L_GAN kao adversarial gubitak diskriminatora. L_GAN se za svaku skalu skalira sa lambdom, i oba (L_REC
i L_GAN) se dodaju na kumulativni gubitak.

Na kraju se računa gubitak diskriminatora koristeći binarnu unakrsnu entropiju nad realnim i lažnim slikama, sa ciljem pogađanja koja je u
pitanju. Svakih save_every iteracija, generator se čuva kao generator.pth fajl, i trening je moguće ranije prekinuti, kada rezultati postanu
zadovoljavajući.

### Tokenizacija slike:
`tokenize_dataset.py` se može koristiti za kreaciju novog skupa podataka, sačinjenog od indeksa tokena nastalih kvantizacijom originalnog
skupa za obuku. Moguće je promeniti file_path promenljivu u main funkciji da pokazuje na putanju do skupa za obuku, kao i
model.load_state_dict(torch.load('vqvae_v4_2999.pth')) imenom obučenog VQ-VAE fajla.

Kreira se dataloader koji ne meša podatke, već redom prolazi kroz frejmove i čuva indeks trenutno procesiranog videa. Za svaki frejm,
on ga kvantizuje u indekse, a zatim ih pretvori u jednodimenzioni niz, kako bi mogao da se koristi za obuku transformera. Ovaj niz pretvara
u listu, i, ako je frejm deo videa koji se već procesira, produžava kumulativnu listu njegovim indeksima. U suprotnom, ažurira indeks
trenutno procesiranog videa, i dodaje indekse u novu listu. Ovo se radi kako bi se razdvojili različiti videi, i ne bi došlo do naglih prelaza
iz jednog videa u drugi. Na karju, svaka od ovih listi se čuva u tekstualnim fajlovima. Potrebno je napraviti out_dataset folder, u koji će
ovi fajlovi biti izvezeni.

### Izvoz brzine i ugla volana:
`tokenize_other.py`, iako se tako zove, ne vrši nikakvu tokenizaciju. Jedina stvar koju ova skripta radi je da kreira tekstualne fajlove
u koje zapisuje brzine i ugao volana. Ovi fajlovi će se kasnije koristiti za obuku simulatora.

### Simulator:
U simulator folderu se nalazi `transformer.py` fajl, u kojem je, pomoću pytorch transformer enkodera i dekodera implementirana varijanta
transformera. Na ulaz forward funkcije se dovode 'input' kao tenzor indeksa ulaznih tokena slike, ugla volana i brzine, kao i 'context' kao
tenzor tokena naredne slike, koja se predviđa. U telu funkcije se najpre generišu sinusoidalni pozicioni enkodinzi i dodaju na input i
context, a zatim se generiše trougaona kauzalna maska za pažnju transformer dekodera. Nakon što se input propusti kroz enkoder, i dobije
tenzor embedinga, on se, zajedno sa kontekstom i kauzalnom maskom, prosleđuje transformer dekorderu. Konačno, izlaz iz dekodera se propušta
kroz linearni sloj, koji generiše logite za svaki od mogućih izlaznih tokena.

Drugi fajl koji se nalazi u ovom folderu je `dataset.py`. Prilikom inicijalizacije, njemu se, između ostalog, prosleđuje putanja do skupa
podataka izgenerisanog pomoću `tokenize_dataset.py` i `tokenize_other.py` skripti u prethodnim koracima. Ovo je folder koji sadrži .txt fajlove
uglova volana, brzina i tokenizovanih frejmova slika. Prvo što DrivingSimulatorDataset klasa radi prilikom inicijalizacije je dobavljanje
putanja do ovih fajlova i njihovo sortiranje po nazivu. Zatim prolazi kroz sve txt fajlove slika, učitava njihove vrednosti u listu
self.data_img, i proračunava dužinu videa uzimajući u obzir eksponencijalnu memoriju. Eksponencijalna memorija je implementirana tako da se
na ulaz transformer enkodera dovode tokeni svakog exponential_base^i-tog frejma prošlosti, tako se na primer za past_length=3, na ulaz dovode
tokeni frejmova indeksa -1, -4, -16.

Nakon učitavanja slika, učitavaju se uglovi volana i brzine iz preostalih fajlova i pamti se najmanja i najveća vrednost - koji će kasnije
biti upotrebljeni kod kvantizacije.

__getitem__ funkcija vrši dobavljanje ulaza, konteksta i izlaza tako da prati pravila eksponencijalne memorije. Na osnovu ulaznog idx-a,
prolazi se redom kroz dužine videa i pronalazi video_number koji odgovara rednom broju videa koji će biti izabran. Zatim se izračunava
context_idx tako što se na početnu idx vrednost (koji označava najraniji trenutak koji će biti prosleđen ulazu), dodaje broj frejmova koji
treba preskočiti zbog dužine eksponencijalne memorije. U petlji se, zatim, na ulaz dodaju tokenizovane slike koje odgovaraju vremenskim
trenucima koje eksponencijalna memorija bira: context_idx - 1, context_idx - 4, ...

Nakon toga se uzimaju sve brzine i uglovi volana (bez preskoka) i dodaju na listu ulaza.

Kontekst, s druge strane, uzima samo tokene na context_length indeksu, redom: slika, ugao volana, brzina, i na početno mesto se dodaje
start token sa vrednosti 0. Target tenzor se pronalazi na sličan način, osim što mu se ne dodaje start token, i što se one-hot enkoduje.

### Obuka simulatora:
Obuka simulatora se vrši u `train_transformer_vqmsgan.py` skripti. Unutar nje se može videti da se past_length postavlja na 1, što znači
da eksponencijalna memorija praktično nije bila korištena (kao ulaz se koristi samo prethodni frejm, brzina i ugao). Nakon inicijalizacije
osnovnih promenljivih, sledi definisanje arhitekture modela. U fajlu je trenutno implementiran najveći model od 200M parametara, ali je
moguće promenom ovih parametara na vrednosti iz apendiksa rada trenirati modele od 17M i 45M parametara.

Parametar warmup_iters je postavljen na 1000 koraka, što znači da će se na početku obuke, prvih 1000 koraka, stopa učenja linearno
povećavati sa 0.1 * bazne vrednosti, na 1.0 * bazne vrednosti. Nakon toga, sledi kreacija dataset promenljive, dataloadera i transformer
modela, a zatim, učitavanje prethodno obučenih VQ-VAE i MSG-GAN modela. Ukoliko se težine ovih modela ne nalaze u ovom folderu, ili su
drugačije nazvane, potrebno je promeniti putanju prilikom njihovog učitavanja.

Kao funkcija gubitka se koristi unakrsna entropija - čime se model obučava da na osnovu ulaza i konteksta, predvidi sledeći token slike.
Optimizator korišten u ovom radu je AdamW, a model se obučava maksinalni broj koraka dat sa max_steps koji koristi Chinchilla recept:
trenirati model na 20x više podataka nego što ima parametara (jednu epohu). Koristi se kosinusni scheduler za stopu učenja, koji u okviru
max_steps - warmup_iters broja iteracija smanjuje stopu učenja faktorom 10.

U glavnoj petlji za obuku se model poziva za input i context dobavljeni od strane dataloader-a, i dobija se izlaz. Ovom izlaznom tenzoru
se dalje menja oblik tako da bude istog formata kao one-hot target tenzor. Nakon toga se izdvajaju izlazi vezani za tokene slike,
uglova i brzina. Uglovi i brzine se ovde ne koriste, zbog čega se njihov gubitak množi sa nulom, i mogu se ignorisati. Time, konačan
gubitak predstavlja unakrsnu entropiju između izlaznih tenzora i target tenzora. Svakih gradient_accumulation_steps iteracija petlje
se vriši optimizacioni korak, kao i korak scheduler-a.

Svakih generate_frequency koraka, radi praćenja progresa, generiše se 10 frejmova videa, počevši od nasumičnog frejma iz skupa podataka.
Generacija funkcioniše tako što se pozove generate_integer_sequence, koji na osnovu ulaza i startnog tokena kao kontekst, token po token
generiše tokene narednih 10 frejmova. Ovi tokeni se zatim reshape-uju u oblik koji odgovara ulazu VQ-VAE kvantizator, koji indekse pretvara
u tenzor embedinga. Ovo dalje može služiti kao ulaz u MSG-GAN, koji na izlazu daje 3 slike različitih rezolucija. Ovde se bira druga slika,
rezolucije 272x68, na nju se docrtava volan zakrivljen za ugao, kao i natpis brzine. Konačno, frejm se čuva u generated_images folder.
Svakih save_every_n_iterations iteracija, čuva se model, zajedno sa ostalim vrednostima, kao što je stanje optimizatora.

### Testiranje simulatora:
Pokretanjem `measurement_test.py` skripte pokreće se faza zaključivanja nad test skupu sa ulaznim vrednostima brzine određenim sa speed=15.0
i angle=0.0. Ova skripta radi realativno neizmenjeno u odnosu na periodično generisanje slika u fazi obuke, s tim su sada brzina i ugao
volana fiksirani na konkretnu vrednost. Kao i kod ostalih skripti, moguće je promeniti putanju do test skupa, VQ-VAE i MSG-GAN modela, kao
i putanju do foldera za izvoz. Promenom parametara sample_size, može se promeniti broj generisanih videa, dok se promenom parametra
video_length može menjati broj frejmova koji će biti generisani. Glavna razlika ovog fajla u odnosu na skriptu za obuku je to što ne koristi
`dataset.py` kao klasu baze podataka, već image_h5_dataset i dobaljenu sliku najpre tokenizuje pozivom tokenize_image() funkcije.

### Računica FID skora:
U fid_calculation folderu se nalazi 1image_dataset_fid.py1 skripta, koja, slično kao i kod prethodnih fajlova, pristupa h5 fajlu, i vraća
sliku za dati broj indeksa. Jedina razlika ovde je ta što se pre vraćanja, rezolucija slike prepolovljava kako bi se poklopila sa rezolucijom
drugog izlaza iz MSG-GAN-a.

Unutar 1fid_calculation.py1 fajla najpre se definise GeneratedImageDataset klasa, koja služi kao skup podataka za dataloader. Ona učitava
sve slike sa sufiksom <broj_frejma>.png, i može se koristiti za učitavanje prethodno generisanih slika pozivom measurment_test.py skripte,
gde broj_frejma označava redni broj frejma za koji se računa FID skor. Promenom generated_folder_path promenljive, može se promeniti putanja
do foldera sa generisanim slikama, dok se promenom real_folder_path promenljive može promeniti putanja do testnog h5 fajla.
InceptionFeatureExtractor klasa kreira model koji preuzima arhitekturu i težine inception_v3 modela, ali bez izlaznih slojeva, tako da se
mogu koristiti obeležja pretposlednjeg linearnog sloja. Funkcija calculate_fid uzima inception_v3 obeležja pravih i generisanih slika, i
računa FID skor po formuli (6) datoj u radu: https://arxiv.org/pdf/1706.08500

U kodu se procena srednje vrednosti inception v3 obeležja na realnim slikama procenjuje na osnovu 1000 nasumično uzrokovanih slika. Kada se
jednom izračunaju ova obeležja, ona bivaju sačuvana u real_features.npy fajl, pa se kasnije mogu ponovo koristiti promenom load_mode=True.
Nakon ovoga, vrši se 100 bootstrap iteracija uzrokovanja generisanog skupa, tako da se mogu odrediti granice pouzdanosti. Ovo se može
ponoviti za svaki broj_frejma.