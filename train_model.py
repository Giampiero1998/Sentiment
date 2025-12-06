import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle
import mlflow
import os

# 1. Configurazione MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Sentiment_Analysis_Production")

# 2. Dataset realistico e bilanciato
data = {
    'text': [
        # -SENTIMENT POSITIVI
        
        # E-commerce / Prodotti
        "Ordered this headset last week and honestly it exceeded my expectations. Sound quality is crisp, battery lasts forever, and the noise cancellation actually works during my commute. Worth every penny!",
        "My daughter absolutely loves this toy! It's been three months and it still works perfectly. The build quality is solid and it keeps her entertained for hours. Definitely buying more from this brand.",
        "Finally found a wireless charger that doesn't overheat my phone. It's fast, looks sleek on my desk, and the LED indicator is subtle enough not to bother me at night. Highly recommended!",
        "This laptop bag is a game changer. Fits my 15 inch MacBook perfectly, has tons of pockets for organizing cables, and the padding makes me feel safe carrying expensive equipment. Great purchase!",
        "Best coffee maker I've owned. Brews in under 5 minutes, coffee tastes fresh every time, and cleaning is super easy. After 6 months of daily use, still working like new.",
        
        # Ristoranti / Food delivery
        "Had dinner here last night with my family and wow, just wow. The pasta was handmade, the tiramisu melted in my mouth, and our waiter Giuseppe was incredibly attentive. Can't wait to come back!",
        "Ordered sushi delivery for the first time from this place. Arrived 20 minutes early, still cold, and the presentation was restaurant-quality. The spicy tuna roll is now my go-to. Will order again for sure!",
        "This bakery makes the best croissants outside of France, no exaggeration. Flaky, buttery, and they're still warm when you buy them in the morning. Lines are long but totally worth the wait.",
        "Grabbed a quick lunch here today. The sandwiches are huge, fresh ingredients, and reasonably priced for the area. Staff was friendly even during the lunch rush. New favorite spot!",
        
        # Hotel / Viaggi
        "Just checked out after a week stay. Room was spotless, bed incredibly comfortable, and the location was perfect for exploring downtown. The breakfast buffet had great variety. Would definitely book again!",
        "This Airbnb was exactly as pictured, which is rare. Host left welcome snacks, WiFi was fast, and the neighborhood felt safe. Honestly one of the best stays I've had. Five stars well deserved!",
        "Took a guided tour with this company and our guide Maria was phenomenal. She knew so much history, took great photos for us, and even recommended amazing local restaurants. Best tour experience ever!",
        
        # Servizi / Software / App
        "Been using this project management tool for 3 months now and my team's productivity has noticeably improved. Interface is intuitive, integrations work seamlessly, and customer support responds within hours.",
        "This meditation app actually helped me with my anxiety. The sessions are the perfect length, the instructor's voice is calming, and I love the sleep stories. Money well spent on the premium subscription.",
        "Switched to this VPN last month and I'm impressed. Connection speeds are fast, never drops, and I can access content from different regions easily. Great value for the price!",
        "This photo editing software is incredible. Learning curve was minimal, tons of filters and tools, and it doesn't slow down my computer. As a hobbyist photographer, it's perfect for my needs.",
        
        # Customer Service
        "Had an issue with my order and contacted support. They responded in 30 minutes, were super apologetic, and sent a replacement with expedited shipping at no cost. That's how you handle customer service!",
        "The team at this car rental agency went above and beyond. They upgraded us for free when they saw we had a baby, showed us how all the features worked, and checkout was a breeze. Impressed!",
        
        # Prodotti di bellezza / Wellness
        "This moisturizer has transformed my skin. After two weeks, my face feels hydrated, looks brighter, and my makeup applies so much smoother. Plus it doesn't have that weird chemical smell. Love it!",
        "Started taking these vitamins a month ago and I genuinely have more energy throughout the day. No weird aftertaste, easy to swallow, and the bottle lasts forever. Definitely repurchasing!",
        
        # Servizi professionali
        "Just had my taxes done by this accountant and I couldn't be happier. He found deductions I didn't know about, explained everything clearly, and saved me a ton of money. Worth every dollar!",
        "Our wedding photographer was amazing. She captured every moment perfectly, made us feel comfortable all day, and delivered the photos ahead of schedule. We're obsessed with how they came out!",
        
        # Negozi fisici
        "This bookstore is a hidden gem. Staff actually reads and gives great recommendations, cozy reading nooks everywhere, and they host fun author events. Supporting local businesses like this feels good!",
        "Went to this hardware store looking for a specific part and the employee spent 15 minutes helping me find exactly what I needed. No pushy sales tactics, just genuine helpfulness. Refreshing!",
        
        # Fitness / Sport
        "Joined this gym two months ago and it's been fantastic. Equipment is always clean and available, classes are challenging but fun, and the trainers are knowledgeable and motivating. Best decision!",
        "These running shoes are perfect. After 100 miles they still have great cushioning, my feet don't hurt anymore, and they're surprisingly lightweight. Worth the investment for serious runners!",
        
        # Intrattenimento
        "This board game was a hit at our family gathering. Easy to learn, fun for all ages, and games don't drag on forever. We've played it five times already and it's still entertaining!",
        "Went to see this movie last night and it was brilliant. Great story, amazing cinematography, and the acting was top notch. Haven't seen something this good in theaters in years!",
        
        # Pet products
        "My cat is obsessed with this toy. It's been three weeks of daily play and it's still in one piece, which says a lot. She's more active now and I love watching her play. Great quality!",
        "This dog food made such a difference. My lab's coat is shinier, he has more energy on walks, and his digestion issues cleared up. Vet even commented on how healthy he looks!",
        
        # Home / Garden
        "These kitchen knives are sharp, balanced, and make meal prep so much faster. After months of use they still hold their edge perfectly. Professional quality at a reasonable price!",
        "Planted these seeds in my garden and the germination rate was incredible. Beautiful flowers bloomed within weeks and they're still going strong. Best purchase for my garden this year!",
        
        # Tech / Gadgets avanzati
        "This smart home hub integrated with everything I own effortlessly. Setup took 10 minutes, the app is super user friendly, and now I can control my whole house from my phone. Living in the future!",
        "Bought this wireless mouse for work and my wrist pain is gone. Ergonomic design is comfortable for long sessions, battery lasts weeks, and the tracking is precise. Should have bought this sooner!",
        
        # Corsi / Formazione
        "Completed this online course last month and learned so much. Instructor was engaging, assignments were practical, and I'm already applying the skills at work. Certificate looks great on LinkedIn too!",
        "This cooking class was so much fun. Chef was patient with beginners, we learned actual techniques, and got to eat everything we made. Taking another class next month for sure!",
        
        # Auto / Moto
        "Took my car here for service and they were honest about what needed fixing immediately versus what could wait. Fair pricing, finished on time, and my car runs smoother. Found my new mechanic!",
        "These car floor mats are built to last. They're waterproof, easy to clean, and fit my car perfectly. After a winter of salty slush, they still look brand new. Quality product!",
        
        # Baby / Kids
        "This baby monitor gives me such peace of mind. Video quality is crystal clear even at night, the app is reliable, and the two-way audio actually works well. Essential for new parents!",
        "My toddler loves these building blocks. They're safe, colorful, and durable enough to survive constant dropping and chewing. Great for developing motor skills and imagination!",
        
        # Fashion / Abbigliamento
        "This jacket is perfect for the unpredictable weather here. Waterproof, breathable, and stylish enough to wear casually. After a month of daily wear, still looks brand new. Fantastic quality!",
        "These jeans fit like they were tailored for me. Comfortable all day, the denim quality is excellent, and they haven't faded after multiple washes. Best jeans I've bought in years!",
        
        # Extra positive reviews (per arrivare a 60)
        "The customer service rep went out of her way to resolve my issue. She was patient, knowledgeable, and genuinely cared about helping me. Companies need more employees like her!",
        "This vacuum cleaner is powerful yet surprisingly quiet. Picks up pet hair effortlessly, easy to empty, and the battery lasts long enough to clean my whole apartment. Great investment!",
        "Attended this concert last weekend and it was unforgettable. Sound quality was perfect, venue was well organized, and the artist gave 110%. One of the best live shows I've seen!",
        "This protein powder actually tastes good, which is rare. Mixes smoothly, no weird aftertaste, and I've noticed better recovery after workouts. Finally found one I'll stick with!",
        "The dentist here is so gentle. First time I've had a cavity filled without anxiety. Staff is friendly, office is clean and modern, and they actually run on schedule. Highly recommend!",
        "This standing desk has helped my back pain so much. Smooth height adjustment, sturdy construction, and plenty of space for dual monitors. Should have bought this years ago!",
        "Got my haircut here yesterday and I'm thrilled with the result. Stylist listened to what I wanted, gave great suggestions, and the price was very reasonable. Found my new salon!",
        "This puzzle was challenging but not frustratingly hard. Quality pieces that fit together perfectly, beautiful image, and great for relaxing evenings. Already ordered another one!",
        "The plumber arrived on time, fixed the leak quickly, cleaned up after himself, and charged exactly what was quoted. Professional service that's hard to find these days!",
        "This portable charger has saved me so many times. Charges my phone three times fully, compact enough for my pocket, and charges quickly itself. Essential for travel!",
        "The language learning app actually works. After three months I can hold basic conversations, lessons are engaging and bite-sized, and the speech recognition is surprisingly accurate!",
        "This mattress completely changed my sleep quality. Perfect firmness, no more back pain, and I wake up actually feeling rested. Best investment I've made for my health!",
        "The bike shop mechanic explained exactly what my bike needed and why. Fair pricing, quality parts, and now it rides like new. Great to find an honest, skilled mechanic!",
        "This sunscreen doesn't leave a white cast and actually stays on during swimming. Not greasy, no chemical smell, and my skin hasn't burned once. Best sunscreen I've tried!",
        "The escape room was so clever and fun. Puzzles were challenging but fair, game master gave perfect hints, and we just barely escaped in time. Can't wait to try another!",
        "This blender is a beast. Crushes ice perfectly for smoothies, easy to clean, and surprisingly quiet for how powerful it is. Makes healthy eating so much easier!",
        
        # -SENTIMENT NEGATIVI
        
        # E-commerce / Prodotti
        "Received this supposedly 'premium' keyboard today and I'm disappointed. Keys feel cheap and mushy, RGB lighting is uneven, and it's already double-typing on some keys. Returning it tomorrow.",
        "This bluetooth speaker died after two weeks. Two weeks! Sound quality was mediocre at best and now it won't even charge. Cheap Chinese garbage. Don't waste your money like I did.",
        "The product photos made this look way better than it actually is. Material is flimsy, stitching is coming apart already, and it's much smaller than expected. Classic case of false advertising.",
        "Ordered this charging cable and it broke within three days of normal use. The wire frayed near the connector and now it's useless. Dollar store cables last longer than this junk.",
        "This 'ergonomic' mouse gave me worse wrist pain than my old one. Design is awkward, buttons are stiff, and it disconnects randomly. Ergonomic my foot. Total waste of fifty bucks.",
        
        # Ristoranti / Food delivery  
        "Waited over two hours for cold pizza that tasted like cardboard. Called three times for updates and got hung up on twice. Never ordering from here again, absolutely terrible service.",
        "This restaurant used to be good but quality has seriously gone downhill. My pasta was overcooked and bland, service was slow despite being half empty, and they got our order wrong. Very disappointing.",
        "Food delivery arrived 90 minutes late and completely cold. Half the order was missing and the driver was rude when I asked about it. Requested a refund and they're ignoring my emails.",
        "Found a hair in my salad. When I politely told the server, they acted like I was the problem and barely apologized. Absolutely disgusting and the manager didn't even come talk to us.",
        
        # Hotel / Viaggi
        "This hotel room was nothing like the photos. Dirty carpet, bathroom hadn't been cleaned properly, and the AC was broken during a heatwave. Front desk was unhelpful. Checked out early.",
        "Booked this Airbnb and showed up to find it was actually someone's basement with mold smell and a broken lock. Host stopped responding to messages. Worst travel experience ever.",
        "Tour guide showed up 45 minutes late, clearly didn't want to be there, and rushed through everything. Couldn't hear half of what he said. Seventy euros completely wasted.",
        
        # Servizi / Software / App
        "This app crashes constantly and customer support is non-existent. Been trying to get a refund for three weeks with zero response. The whole thing feels like a scam.",
        "Paid for premium subscription and half the features don't work. Interface is clunky and confusing. Cancelled within a week but they won't refund me. Stay away from this garbage.",
        "This software bricked my computer. Had to do a full system restore and lost hours of work. Their 'support team' just sent generic copy-paste responses. Absolutely infuriating.",
        "The VPN constantly disconnects and speeds are painfully slow. Contacted support multiple times and got useless troubleshooting steps that didn't help. Such a ripoff.",
        
        # Customer Service
        "Worst customer service I've ever experienced. Been trying to return a defective item for two weeks and keep getting bounced between departments. No one takes responsibility.",
        "They charged my card twice and refuse to refund the duplicate charge. Been on hold for hours multiple times. This is borderline theft and I'm considering legal action.",
        
        # Prodotti di bellezza / Wellness
        "This moisturizer broke me out horribly. Face is covered in painful cystic acne now. The 'natural ingredients' claim is clearly BS. Threw it in the trash after three days.",
        "These vitamins made me nauseous every single morning. Couldn't keep taking them. Customer service said no refunds on opened bottles even though product made me sick. Terrible policy.",
        
        # Servizi professionali  
        "This accountant made several mistakes on my tax return that I caught myself. When I questioned him he got defensive. Had to hire someone else to fix his mess. Completely incompetent.",
        "Wedding photographer was late, took blurry photos, and delivered them four months after the wedding. Half are unusable. Paid thousands for amateur work. Biggest regret of our wedding.",
        
        # Negozi fisici
        "Employee at this store followed me around like I was going to steal something. Made me extremely uncomfortable. When I asked for help they were dismissive and rude. Never going back.",
        "Went to this hardware store for a simple part and the worker gave me completely wrong information. Had to make three trips because they kept selling me the wrong items. Incompetent staff.",
        
        # Fitness / Sport
        "This gym is always overcrowded, equipment is constantly broken, and the locker room is disgusting. Management doesn't care about complaints. Cancelling my membership immediately.",
        "These running shoes gave me blisters on the first run. The sizing is completely off and the cushioning is non-existent. My cheap old shoes were more comfortable than these.",
        
        # Intrattenimento
        "This board game has terrible instructions that make no sense. Rules are unclear and the game drags on forever. We quit halfway through out of boredom. What a disappointment.",
        "That movie was two hours of my life I'll never get back. Plot made no sense, acting was wooden, and the ending was ridiculous. Don't believe the hype, it's terrible.",
        
        # Pet products
        "This pet toy broke apart in minutes and my dog tried to swallow the pieces. Extremely dangerous. Could have choked him. This shouldn't be sold, it's a safety hazard.",
        "This dog food made my pet sick immediately. He had diarrhea for days. Vet bills cost more than the food. Check the ingredients before buying, it's full of fillers and junk.",
        
        # Home / Garden
        "These kitchen knives are dull right out of the box. Can't even cut a tomato properly. The handles feel cheap and hollow. Total garbage that belongs in the dollar store.",
        "Not a single seed from this packet germinated. Complete waste of money and planting season. Either the seeds are old or defective. Very frustrating for a gardener.",
        
        # Tech / Gadgets
        "Smart home hub doesn't work with half my devices despite claiming compatibility. Setup was a nightmare, keeps losing connection, and the app is buggy. Returning this paperweight.",
        "This wireless mouse stops working randomly and I have to reset it constantly. Battery dies in days, not weeks like advertised. Cheap plastic construction feels like it'll break any second.",
        
        # Corsi / Formazione
        "This online course is completely outdated. Information is from five years ago and no longer relevant. Videos are low quality and instructor is monotone and boring. Total waste of money.",
        "Cooking class was chaotic and disorganized. Chef was unprepared, ingredients were missing, and everything tasted terrible. Can't believe I paid for this disaster of an experience.",
        
        # Auto / Moto
        "This mechanic charged me for repairs they didn't do. My car has the same problem it went in with. When I confronted them they denied everything. These people are dishonest crooks.",
        "These car mats don't fit properly and slide around while driving, which is dangerous. Material is thin and cheap. Already falling apart after two weeks. Horrible quality.",
        
        # Baby / Kids  
        "This baby monitor constantly loses signal and the video quality is grainy and useless at night. App crashes frequently. Can't trust it for my baby's safety. Returning immediately.",
        "These toys broke within hours and have sharp edges that cut my child's finger. Poor quality control and dangerous design. How did these pass safety standards?",
        
        # Fashion / Abbigliamento
        "This jacket is not waterproof at all despite advertising. Got soaked in light rain. Seams are already coming apart and color is fading. Cheaply made garbage, don't buy it.",
        "These jeans shrank two sizes after one wash following care instructions exactly. Dye bled everywhere and ruined other clothes. Quality is abysmal for the price. Very angry.",
        
        # Extra negative reviews (per arrivare a 60)
        "Customer service rep was incredibly rude and condescending. Made me feel stupid for asking questions. Hung up on me when I asked for a manager. Absolutely unacceptable behavior.",
        "This vacuum broke after one month. Motor burned out and smells like burning plastic. Company won't honor warranty and support is unreachable. Avoid this brand entirely.",
        "Concert was a disaster. Sound system kept cutting out, couldn't see anything from our expensive seats, and artist performed for only 45 minutes. Complete ripoff, want my money back.",
        "This protein powder clumps no matter how much I shake it and tastes like chemical sweetener. Made me feel sick to my stomach. Can't finish the container, it's vile.",
        "Dentist was rough and it hurt way more than it should have. Kept trying to upsell unnecessary procedures. Office is dingy and equipment looks ancient. Finding a new dentist.",
        "This standing desk wobbles terribly at standing height and the motor makes loud grinding noises. Paint is chipping already. Poor quality for the premium price tag.",
        "Got the worst haircut of my life here. Stylist completely ignored what I asked for and butchered my hair. Now I have to wear hats until it grows out. Absolutely awful.",
        "This puzzle was missing three pieces. Spent hours on it only to find it's incomplete. Company won't send replacements. How is quality control this bad? So frustrating.",
        "Plumber was two hours late, did a sloppy job, and now the leak is worse than before. Charged me premium rates for amateur work. Have to call someone else to fix his mistakes.",
        "Portable charger exploded while charging and burned my desk. Could have started a fire. This is incredibly dangerous and should be recalled immediately. Terrifying experience.",
        "Language app is full of errors and teaches incorrect grammar. Native speakers in the forum complain constantly. Waste of subscription money, better free alternatives exist.",
        "This mattress is horribly uncomfortable and sags in the middle already after two months. Back pain is worse than ever. Return policy is a scam with huge fees. Stuck with this nightmare.",
        "Bike shop mechanic did unnecessary work I didn't authorize and charged me double the estimate. My bike still makes the same noise. They're dishonest and incompetent.",
        "This sunscreen burned my skin badly. Left red irritated patches all over. Contains harsh chemicals despite 'gentle formula' claim. Had to see a dermatologist. Dangerous product.",
        "Escape room was boring with obvious puzzles and broken props. Game master was on his phone the whole time and hints were useless. Felt like a cash grab. Not worth it.",
        "Blender motor died after a week of smoothies. Blade assembly is cheap plastic that cracked. Makes horrible grinding noise. Kitchen is full of blender smell. Terrible appliance.",
    ],
    'sentiment': [1]*60 + [0]*60  # 60 positivi, 60 negativi - dataset bilanciato e realistico
}

df = pd.DataFrame(data)

# Parametri del modello ottimizzati
MAX_FEATURES = 300  # Aumentato per catturare più pattern linguistici
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 20% per test, 80% per training
C_REGULARIZATION = 1.0  # Parametro di regolarizzazione
MIN_F1_SCORE_THRESHOLD = 0.85

# Inizio della run MLflow
with mlflow.start_run() as run:
    
    # Log dei parametri
    mlflow.log_param("language", "english")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("dataset_size", len(df))
    mlflow.log_param("C_regularization", C_REGULARIZATION)
    mlflow.log_param("positive_samples", sum(df['sentiment'] == 1))
    mlflow.log_param("negative_samples", sum(df['sentiment'] == 0))
    
    # 3. Pre-processing e Addestramento
    X = df['text']
    y = df['sentiment']

    # Vettorizzazione ottimizzata con n-grams
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),  # Usa unigrammi e bigrammi
        min_df=1,
        sublinear_tf=True
    )
    X_vectorized = vectorizer.fit_transform(X)

    # Split stratificato per mantenere bilanciamento
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y
    )

    # Modello con parametri ottimizzati
    model = LogisticRegression(
        max_iter=1000,
        C=C_REGULARIZATION,
        class_weight='balanced',
        solver='liblinear'
    )
    model.fit(X_train, y_train)

    # 4. Valutazione e logging delle metriche
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Log delle metriche di test
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_f1_score", f1)
    
    # Salva l'F1-Score in un file per il Quality Gate
    METRICS_FILE = 'model_metrics.txt'
    with open(METRICS_FILE, 'w') as f:
        f.write(str(f1))
    
    print(f"\n{'='*60}")
    print(f"METRICHE DI VALUTAZIONE DEL MODELLO")
    print(f"{'='*60}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    print(f"{'='*60}\n")
    
    # Quality Gate check
    if f1 >= MIN_F1_SCORE_THRESHOLD:
        print(f"✅ Quality Gate PASSED: F1-Score ({f1:.4f}) >= Threshold ({MIN_F1_SCORE_THRESHOLD})")
    else:
        print(f"❌ Quality Gate FAILED: F1-Score ({f1:.4f}) < Threshold ({MIN_F1_SCORE_THRESHOLD})")

    # 5. Serializzazione e archiviazione locale
    MODEL_PATH = 'sentiment_model.pkl'
    VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

    # Archivia i file .pkl sul filesystem
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"Modello e vettorizzatore salvati e tracciati su MLflow.\n")
    
    # Log dei file come artifact in MLflow
    mlflow.log_artifact(MODEL_PATH)
    mlflow.log_artifact(VECTORIZER_PATH)
    mlflow.log_artifact(METRICS_FILE)