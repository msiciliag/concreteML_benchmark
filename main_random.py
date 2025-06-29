import yaml
import random
import time
import click
import platform
import json
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo, list_available_datasets
from multiprocessing import Pool
from itertools import repeat
import importlib
import os
import msgpack
import mlflow
import importlib.metadata
import hashlib

DEFAULT_RANDOM_SEED = 42
DEFAULT_EXP_NUMBER = 20

def load_progress(progress_file):
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "rb") as f:
                packed_data = f.read()
                unpacked_data = msgpack.unpackb(packed_data, raw=False)
                return set(unpacked_data)
        except Exception as e:
            print(f"Error retrieving progress file: {e}")
            return set()
    return set()

def save_progress(progress, progress_file):
    with open(progress_file, "wb") as f:
        packed_data = msgpack.packb(list(progress), use_bin_type=True)
        f.write(packed_data)

def hash_config(config):
    config_str = str(sorted(config.items()))
    return hashlib.md5(config_str.encode()).hexdigest()

def instantiate_models(model_config: dict, param_config: dict, fhe_config: dict):
    """Creates instances of the models with the given configuration"""
    model_name, model_module = model_config["name"], model_config["module_name"]
    module = importlib.import_module(model_module)
    class_ = getattr(module, model_name)
    instance = class_(**param_config)
    
    fhe_model_name, fhe_model_module = model_config["fhe_name"], model_config["fhe_module_name"]
    module = importlib.import_module(fhe_model_module)
    class_ = getattr(module, fhe_model_name)
    instance_fhe = class_(**param_config, **fhe_config)
    
    return instance, instance_fhe

def expand_config_param(param):
    """Expand a configuration parameter into an iterable"""
    if 'values' in param:
        return param['values']
    if 'min' in param and 'max' in param and 'step' in param:
        return range(param['min'], param['max'], param['step'])
    raise ValueError(f"Unsupported parameter configuration: {param}")

def log_system_info():
    """Log the main specs of the machine"""
    system_info = {
        "platform": platform.system(),
        "platform-release": platform.release(),
        "platform-version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "ram": f"{round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024. ** 3), 2)} GB"
    }
    for key, value in system_info.items():
        mlflow.log_param(key, value)

def log_library_versions():
    """Log the versions of the main libraries used"""
    libraries = ["scikit-learn", "concrete-ml"]
    for library in libraries:
        try:
            version = importlib.metadata.version(library)
            mlflow.log_param(f"{library}_version", version)
        except importlib.metadata.PackageNotFoundError:
            mlflow.log_param(f"{library}_version", "not installed")

def get_random_id():
    """Gets a random id from the list of the avaliable datasets in UCI"""
    #TODO make a request to https://archive.ics.uci.edu/api/datasets/list for getting the list
    json_string = """{"status":200,"statusText":"OK","data":[{"id":1,"name":"Abalone"},{"id":2,"name":"Adult"},{"id":3,"name":"Annealing"},{"id":4,"name":"Anonymous Microsoft Web Data"},{"id":5,"name":"Arrhythmia"},{"id":6,"name":"Artificial Characters"},{"id":7,"name":"Audiology (Original)"},{"id":8,"name":"Audiology (Standardized)"},{"id":9,"name":"Auto MPG"},{"id":10,"name":"Automobile"},{"id":11,"name":"Badges"},{"id":12,"name":"Balance Scale"},{"id":13,"name":"Balloons"},{"id":14,"name":"Breast Cancer"},{"id":15,"name":"Breast Cancer Wisconsin (Original)"},{"id":16,"name":"Breast Cancer Wisconsin (Prognostic)"},{"id":17,"name":"Breast Cancer Wisconsin (Diagnostic)"},{"id":18,"name":"Pittsburgh Bridges"},{"id":19,"name":"Car Evaluation"},{"id":20,"name":"Census Income"},{"id":21,"name":"Chess (King-Rook vs. King-Knight)"},{"id":22,"name":"Chess (King-Rook vs. King-Pawn)"},{"id":23,"name":"Chess (King-Rook vs. King)"},{"id":24,"name":"Chess (Domain Theories)"},{"id":25,"name":"Bach Chorales"},{"id":26,"name":"Connect-4"},{"id":27,"name":"Credit Approval"},{"id":28,"name":"Japanese Credit Screening"},{"id":29,"name":"Computer Hardware"},{"id":30,"name":"Contraceptive Method Choice"},{"id":31,"name":"Covertype"},{"id":32,"name":"Cylinder Bands"},{"id":33,"name":"Dermatology"},{"id":34,"name":"Diabetes"},{"id":35,"name":"DGP2 - The Second Data Generation Program"},{"id":36,"name":"Document Understanding"},{"id":37,"name":"EBL Domain Theories"},{"id":38,"name":"Echocardiogram"},{"id":39,"name":"Ecoli"},{"id":40,"name":"Flags"},{"id":41,"name":"Function Finding"},{"id":42,"name":"Glass Identification"},{"id":43,"name":"Haberman's Survival"},{"id":44,"name":"Hayes-Roth"},{"id":45,"name":"Heart Disease"},{"id":46,"name":"Hepatitis"},{"id":47,"name":"Horse Colic"},{"id":49,"name":"ICU"},{"id":50,"name":"Image Segmentation"},{"id":51,"name":"Internet Advertisements"},{"id":52,"name":"Ionosphere"},{"id":53,"name":"Iris"},{"id":54,"name":"ISOLET"},{"id":55,"name":"Kinship"},{"id":56,"name":"Labor Relations"},{"id":57,"name":"LED Display Domain"},{"id":58,"name":"Lenses"},{"id":59,"name":"Letter Recognition"},{"id":60,"name":"Liver Disorders"},{"id":61,"name":"Logic Theorist"},{"id":62,"name":"Lung Cancer"},{"id":63,"name":"Lymphography"},{"id":64,"name":"Mechanical Analysis"},{"id":65,"name":"Meta-data"},{"id":66,"name":"Mobile Robots"},{"id":67,"name":"Molecular Biology (Promoter Gene Sequences)"},{"id":68,"name":"Molecular Biology (Protein Secondary Structure)"},{"id":69,"name":"Molecular Biology (Splice-junction Gene Sequences)"},{"id":70,"name":"MONK's Problems"},{"id":71,"name":"Moral Reasoner"},{"id":72,"name":"Multiple Features"},{"id":73,"name":"Mushroom"},{"id":74,"name":"Musk (Version 1)"},{"id":75,"name":"Musk (Version 2)"},{"id":76,"name":"Nursery"},{"id":77,"name":"Othello Domain Theory"},{"id":78,"name":"Page Blocks Classification"},{"id":80,"name":"Optical Recognition of Handwritten Digits"},{"id":81,"name":"Pen-Based Recognition of Handwritten Digits"},{"id":82,"name":"Post-Operative Patient"},{"id":83,"name":"Primary Tumor"},{"id":84,"name":"Prodigy"},{"id":85,"name":"Qualitative Structure Activity Relationships"},{"id":86,"name":"Quadruped Mammals"},{"id":87,"name":"Servo"},{"id":88,"name":"Shuttle Landing Control"},{"id":89,"name":"Solar Flare"},{"id":90,"name":"Soybean (Large)"},{"id":91,"name":"Soybean (Small)"},{"id":92,"name":"Challenger USA Space Shuttle O-Ring"},{"id":93,"name":"Low Resolution Spectrometer"},{"id":94,"name":"Spambase"},{"id":95,"name":"SPECT Heart"},{"id":96,"name":"SPECTF Heart"},{"id":97,"name":"Sponge"},{"id":98,"name":"Statlog Project"},{"id":99,"name":"Student Loan Relational"},{"id":100,"name":"Teaching Assistant Evaluation"},{"id":101,"name":"Tic-Tac-Toe Endgame"},{"id":102,"name":"Thyroid Disease"},{"id":103,"name":"Trains"},{"id":104,"name":"University"},{"id":105,"name":"Congressional Voting Records"},{"id":106,"name":"Water Treatment Plant"},{"id":107,"name":"Waveform Database Generator (Version 1)"},{"id":108,"name":"Waveform Database Generator (Version 2)"},{"id":109,"name":"Wine"},{"id":110,"name":"Yeast"},{"id":111,"name":"Zoo"},{"id":112,"name":"Undocumented"},{"id":113,"name":"Twenty Newsgroups"},{"id":114,"name":"Australian Sign Language Signs"},{"id":115,"name":"Australian Sign Language Signs (High Quality)"},{"id":116,"name":"US Census Data (1990)"},{"id":117,"name":"Census-Income (KDD)"},{"id":118,"name":"Coil 1999 Competition Data"},{"id":119,"name":"Corel Image Features"},{"id":120,"name":"E. Coli Genes"},{"id":121,"name":"EEG Database"},{"id":122,"name":"El Nino"},{"id":123,"name":"Entree Chicago Recommendation Data"},{"id":124,"name":"CMU Face Images"},{"id":125,"name":"Insurance Company Benchmark (COIL 2000)"},{"id":126,"name":"Internet Usage Data"},{"id":127,"name":"IPUMS Census Database"},{"id":128,"name":"Japanese Vowels"},{"id":129,"name":"KDD Cup 1998"},{"id":130,"name":"KDD Cup 1999 Data"},{"id":131,"name":"M. Tuberculosis Genes"},{"id":132,"name":"Movie"},{"id":133,"name":"MSNBC.com Anonymous Web Data"},{"id":134,"name":"NSF Research Award Abstracts 1990-2003"},{"id":135,"name":"Pioneer-1 Mobile Robot Data"},{"id":136,"name":"Pseudo Periodic Synthetic Time Series"},{"id":137,"name":"Reuters-21578 Text Categorization Collection"},{"id":138,"name":"Robot Execution Failures"},{"id":139,"name":"Synthetic Control Chart Time Series"},{"id":140,"name":"Syskill and Webert Web Page Ratings"},{"id":141,"name":"UNIX User Data"},{"id":142,"name":"Volcanoes on Venus - JARtool experiment"},{"id":143,"name":"Statlog (Australian Credit Approval)"},{"id":144,"name":"Statlog (German Credit Data)"},{"id":145,"name":"Statlog (Heart)"},{"id":146,"name":"Statlog (Landsat Satellite)"},{"id":147,"name":"Statlog (Image Segmentation)"},{"id":148,"name":"Statlog (Shuttle)"},{"id":149,"name":"Statlog (Vehicle Silhouettes)"},{"id":150,"name":"Connectionist Bench (Nettalk Corpus)"},{"id":151,"name":"Connectionist Bench (Sonar, Mines vs. Rocks)"},{"id":152,"name":"Connectionist Bench (Vowel Recognition - Deterding Data)"},{"id":153,"name":"Economic Sanctions"},{"id":154,"name":"Protein Data"},{"id":155,"name":"Cloud"},{"id":156,"name":"CalIt2 Building People Counts"},{"id":157,"name":"Dodgers Loop Sensor"},{"id":158,"name":"Poker Hand"},{"id":159,"name":"MAGIC Gamma Telescope"},{"id":160,"name":"UJI Pen Characters"},{"id":161,"name":"Mammographic Mass"},{"id":162,"name":"Forest Fires"},{"id":163,"name":"Reuters Transcribed Subset"},{"id":164,"name":"Bag of Words"},{"id":165,"name":"Concrete Compressive Strength"},{"id":166,"name":"Hill-Valley"},{"id":167,"name":"Arcene"},{"id":168,"name":"Dexter"},{"id":169,"name":"Dorothea"},{"id":170,"name":"Gisette"},{"id":171,"name":"Madelon"},{"id":172,"name":"Ozone Level Detection"},{"id":173,"name":"Abscisic Acid Signaling Network"},{"id":174,"name":"Parkinsons"},{"id":175,"name":"Character Trajectories"},{"id":176,"name":"Blood Transfusion Service Center"},{"id":177,"name":"UJI Pen Characters (Version 2)"},{"id":178,"name":"Semeion Handwritten Digit"},{"id":179,"name":"SECOM"},{"id":180,"name":"Plants"},{"id":181,"name":"Libras Movement"},{"id":182,"name":"Concrete Slump Test"},{"id":183,"name":"Communities and Crime"},{"id":184,"name":"Acute Inflammations"},{"id":186,"name":"Wine Quality"},{"id":187,"name":"URL Reputation"},{"id":188,"name":"p53 Mutants"},{"id":189,"name":"Parkinsons Telemonitoring"},{"id":190,"name":"Demospongiae"},{"id":191,"name":"Opinosis Opinion / Review"},{"id":192,"name":"Breast Tissue"},{"id":193,"name":"Cardiotocography"},{"id":194,"name":"Wall-Following Robot Navigation Data"},{"id":195,"name":"Spoken Arabic Digit"},{"id":196,"name":"Localization Data for Person Activity"},{"id":197,"name":"AutoUniv"},{"id":198,"name":"Steel Plates Faults"},{"id":199,"name":"MiniBooNE particle identification"},{"id":203,"name":"Year Prediction MSD"},{"id":204,"name":"PEMS-SF"},{"id":205,"name":"OpinRank Review Dataset"},{"id":206,"name":"Relative location of CT slices on axial axis"},{"id":208,"name":"Online Handwritten Assamese Characters Dataset"},{"id":209,"name":"PubChem Bioassay Data"},{"id":210,"name":"Record Linkage Comparison Patterns"},{"id":211,"name":"Communities and Crime Unnormalized"},{"id":212,"name":"Vertebral Column"},{"id":213,"name":"EMG Physical Action Data Set"},{"id":214,"name":"Vicon Physical Action Data Set"},{"id":215,"name":"Amazon Commerce Reviews"},{"id":216,"name":"Amazon Access Samples"},{"id":217,"name":"Reuter_50_50"},{"id":218,"name":"Farm Ads"},{"id":219,"name":"DBWorld e-mails"},{"id":220,"name":"KEGG Metabolic Relation Network (Directed)"},{"id":221,"name":"KEGG Metabolic Reaction Network (Undirected)"},{"id":222,"name":"Bank Marketing"},{"id":223,"name":"YouTube Comedy Slam Preference Data"},{"id":224,"name":"Gas Sensor Array Drift Dataset"},{"id":225,"name":"ILPD (Indian Liver Patient Dataset)"},{"id":226,"name":"OPPORTUNITY Activity Recognition"},{"id":227,"name":"Nomao"},{"id":228,"name":"SMS Spam Collection"},{"id":229,"name":"Skin Segmentation"},{"id":230,"name":"Planning Relax"},{"id":231,"name":"PAMAP2 Physical Activity Monitoring"},{"id":232,"name":"Restaurant & consumer data"},{"id":233,"name":"CNAE-9"},{"id":235,"name":"Individual Household Electric Power Consumption"},{"id":236,"name":"Seeds"},{"id":237,"name":"Northix"},{"id":238,"name":"QtyT40I10D100K"},{"id":239,"name":"Legal Case Reports"},{"id":240,"name":"Human Activity Recognition Using Smartphones"},{"id":241,"name":"One-hundred plant species leaves data set"},{"id":242,"name":"Energy Efficiency"},{"id":243,"name":"Yacht Hydrodynamics"},{"id":244,"name":"Fertility"},{"id":245,"name":"Daphnet Freezing of Gait"},{"id":246,"name":"3D Road Network (North Jutland, Denmark)"},{"id":247,"name":"ISTANBUL STOCK EXCHANGE"},{"id":248,"name":"Buzz in social media "},{"id":249,"name":"First-order theorem proving"},{"id":250,"name":"Wearable Computing: Classification of Body Postures and Movements (PUC-Rio)"},{"id":251,"name":"Gas sensor arrays in open sampling settings"},{"id":252,"name":"Climate Model Simulation Crashes"},{"id":253,"name":"MicroMass"},{"id":254,"name":"QSAR biodegradation"},{"id":255,"name":"BLOGGER"},{"id":256,"name":"Daily and Sports Activities"},{"id":257,"name":"User Knowledge Modeling"},{"id":259,"name":"Reuters RCV1 RCV2 Multilingual, Multiview Text Categorization Test collection"},{"id":260,"name":"NYSK"},{"id":262,"name":"Turkiye Student Evaluation"},{"id":263,"name":"ser Knowledge Modeling Data (Students' Knowledge Levels on DC Electrical Machines)"},{"id":264,"name":"EEG Eye State"},{"id":265,"name":"Physicochemical Properties of Protein Tertiary Structure"},{"id":266,"name":"seismic-bumps"},{"id":267,"name":"Banknote Authentication"},{"id":268,"name":"USPTO Algorithm Challenge, run by NASA-Harvard Tournament Lab and TopCoder    Problem: Pat"},{"id":269,"name":"YouTube Multiview Video Games Dataset"},{"id":270,"name":"Gas Sensor Array Drift at Different Concentrations"},{"id":271,"name":"Activities of Daily Living (ADLs) Recognition Using Binary Sensors"},{"id":272,"name":"SkillCraft1 Master Table Dataset"},{"id":273,"name":"Weight Lifting Exercises monitored with Inertial Measurement Units"},{"id":274,"name":"SML2010"},{"id":275,"name":"Bike Sharing"},{"id":276,"name":"Predict keywords activities in a online social media"},{"id":277,"name":"Thoracic Surgery Data"},{"id":278,"name":"EMG dataset in Lower Limb"},{"id":279,"name":"SUSY"},{"id":280,"name":"HIGGS"},{"id":281,"name":"Qualitative_Bankruptcy"},{"id":282,"name":"LSVT Voice Rehabilitation"},{"id":283,"name":"Dataset for ADL Recognition with Wrist-worn Accelerometer"},{"id":285,"name":"Wilt"},{"id":286,"name":"User Identification From Walking Activity"},{"id":287,"name":"Activity Recognition from Single Chest-Mounted Accelerometer"},{"id":288,"name":"Leaf"},{"id":289,"name":"Dresses_Attribute_Sales"},{"id":290,"name":"Tamilnadu Electricity Board Hourly Readings"},{"id":291,"name":"Airfoil Self-Noise"},{"id":292,"name":"Wholesale customers"},{"id":293,"name":"Twitter Data set for Arabic Sentiment Analysis"},{"id":294,"name":"Combined Cycle Power Plant"},{"id":295,"name":"Urban Land Cover"},{"id":296,"name":"Diabetes 130-US Hospitals for Years 1999-2008"},{"id":298,"name":"Bach Choral Harmony"},{"id":299,"name":"StoneFlakes"},{"id":300,"name":"Tennis Major Tournament Match Statistics"},{"id":301,"name":"Parkinson's Speech with Multiple Types of Sound Recordings"},{"id":302,"name":"Gesture Phase Segmentation"},{"id":303,"name":"Perfume Data"},{"id":304,"name":"BlogFeedback"},{"id":305,"name":"REALDISP Activity Recognition Dataset"},{"id":306,"name":"Newspaper and magazine images segmentation dataset"},{"id":307,"name":"AAAI 2014 Accepted Papers"},{"id":308,"name":"Gas sensor array under flow modulation"},{"id":309,"name":"Gas sensor array exposed to turbulent gas mixtures"},{"id":310,"name":"UJIIndoorLoc"},{"id":311,"name":"Sentence Classification"},{"id":312,"name":"Dow Jones Index"},{"id":313,"name":"sEMG for Basic Hand movements"},{"id":314,"name":"AAAI 2013 Accepted Papers"},{"id":315,"name":"Geographical Origin of Music"},{"id":316,"name":"Condition Based Maintenance of Naval Propulsion Plants"},{"id":317,"name":"Grammatical Facial Expressions"},{"id":318,"name":"NoisyOffice"},{"id":319,"name":"MHEALTH"},{"id":320,"name":"Student Performance"},{"id":321,"name":"ElectricityLoadDiagrams20112014"},{"id":322,"name":"Gas sensor array under dynamic gas mixtures"},{"id":323,"name":"microblogPCU"},{"id":324,"name":"Firm-Teacher_Clave-Direction_Classification"},{"id":325,"name":"Dataset for Sensorless Drive Diagnosis"},{"id":326,"name":"TV News Channel Commercial Detection Dataset"},{"id":327,"name":"Phishing Websites"},{"id":328,"name":"Greenhouse Gas Observing Network"},{"id":329,"name":"Diabetic Retinopathy Debrecen"},{"id":330,"name":"HIV-1 protease cleavage"},{"id":331,"name":"Sentiment Labelled Sentences"},{"id":332,"name":"Online News Popularity"},{"id":333,"name":"Forest type mapping"},{"id":334,"name":"wiki4HE"},{"id":335,"name":"Online Video Characteristics and Transcoding Time Dataset"},{"id":336,"name":"Chronic Kidney Disease"},{"id":337,"name":"Machine Learning based ZZAlpha Ltd. Stock Recommendations 2012-2014"},{"id":338,"name":"Folio"},{"id":339,"name":"Taxi Service Trajectory - Prediction Challenge, ECML PKDD 2015"},{"id":340,"name":"Cuff-Less Blood Pressure Estimation"},{"id":341,"name":"Smartphone-Based Recognition of Human Activities and Postural Transitions"},{"id":342,"name":"Mice Protein Expression"},{"id":343,"name":"UJIIndoorLoc-Mag"},{"id":344,"name":"Heterogeneity Activity Recognition"},{"id":346,"name":"Educational Process Mining (EPM): A Learning Analytics Data Set"},{"id":347,"name":"HEPMASS"},{"id":348,"name":"Indoor User Movement Prediction from RSS data"},{"id":349,"name":"Open University Learning Analytics dataset"},{"id":350,"name":"Default of Credit Card Clients"},{"id":351,"name":"Mesothelioma's disease data set "},{"id":352,"name":"Online Retail"},{"id":353,"name":"SIFT10M"},{"id":354,"name":"GPS Trajectories"},{"id":355,"name":"Detect Malacious Executable(AntiVirus)"},{"id":357,"name":"Occupancy Detection "},{"id":358,"name":"Improved Spiral Test Using Digitized Graphics Tablet for Monitoring Parkinson's Disease"},{"id":359,"name":"News Aggregator"},{"id":360,"name":"Air Quality"},{"id":361,"name":"Twin gas sensor arrays"},{"id":362,"name":"Gas sensors for home activity monitoring"},{"id":363,"name":"Facebook Comment Volume"},{"id":364,"name":"Smartphone Dataset for Human Activity Recognition (HAR) in Ambient Assisted Living (AAL)"},{"id":365,"name":"Polish Companies Bankruptcy"},{"id":366,"name":"Activity Recognition system based on Multisensor data fusion (AReM)"},{"id":367,"name":"Dota2 Games Results"},{"id":368,"name":"Facebook Metrics"},{"id":369,"name":"UbiqLog (smartphone lifelogging)"},{"id":371,"name":"NIPS Conference Papers 1987-2015"},{"id":372,"name":"HTRU2"},{"id":373,"name":"Drug Consumption (Quantified)"},{"id":374,"name":"Appliances Energy Prediction"},{"id":375,"name":"Miskolc IIS Hybrid IPS"},{"id":376,"name":"KDC-4007 dataset Collection"},{"id":377,"name":"Geo-Magnetic field and WLAN dataset for indoor localisation from wristband and smartphone"},{"id":378,"name":"DrivFace"},{"id":379,"name":"Website Phishing"},{"id":380,"name":"YouTube Spam Collection"},{"id":381,"name":"Beijing PM2.5"},{"id":382,"name":"Cargo 2000 Freight Tracking and Tracing"},{"id":383,"name":"Cervical Cancer (Risk Factors)"},{"id":384,"name":"Quality Assessment of Digital Colposcopies"},{"id":385,"name":"KASANDR"},{"id":386,"name":"FMA: A Dataset For Music Analysis"},{"id":387,"name":"Air quality"},{"id":389,"name":"Devanagari Handwritten Character Dataset"},{"id":390,"name":"Stock Portfolio Performance"},{"id":391,"name":"MoCap Hand Postures"},{"id":392,"name":"Early biomarkers of Parkinson’s disease based on natural connected speech"},{"id":393,"name":"Data for Software Engineering Teamwork Assessment in Education Setting"},{"id":394,"name":"PM2.5 Data of Five Chinese Cities"},{"id":395,"name":"Parkinson Disease Spiral Drawings Using Digitized Graphics Tablet"},{"id":396,"name":"Sales Transactions Weekly"},{"id":397,"name":"Las Vegas Strip"},{"id":398,"name":"Eco-hotel"},{"id":399,"name":"MEU-Mobile KSD"},{"id":400,"name":"Crowdsourced Mapping"},{"id":401,"name":"gene expression cancer RNA-Seq"},{"id":402,"name":"Hybrid Indoor Positioning Dataset from WiFi RSSI, Bluetooth and magnetometer"},{"id":403,"name":"chestnut - LARVIC"},{"id":404,"name":"Burst Header Packet (BHP) flooding attack on Optical Burst Switching (OBS) Network"},{"id":405,"name":"Motion Capture Hand Postures"},{"id":406,"name":"Anuran Calls (MFCCs)"},{"id":407,"name":"TTC-3600: Benchmark dataset for Turkish text categorization"},{"id":408,"name":"Gastrointestinal Lesions in Regular Colonoscopy"},{"id":409,"name":"Daily Demand Forecasting Orders"},{"id":410,"name":"Paper Reviews"},{"id":411,"name":"extention of Z-Alizadeh sani dataset"},{"id":412,"name":"Z-Alizadeh Sani"},{"id":413,"name":"Dynamic Features of VirusShare Executables"},{"id":414,"name":"IDA2016Challenge"},{"id":415,"name":"DSRC Vehicle Communications"},{"id":416,"name":"Mturk User-Perceived Clusters over Images"},{"id":417,"name":"Character Font Images"},{"id":418,"name":"DeliciousMIL: A Data Set for Multi-Label Multi-Instance Learning with Instance Labels"},{"id":419,"name":"Autistic Spectrum Disorder Screening Data for Children  "},{"id":420,"name":"Autistic Spectrum Disorder Screening Data for Adolescent   "},{"id":421,"name":"APS Failure at Scania Trucks"},{"id":422,"name":"Wireless Indoor Localization"},{"id":423,"name":"HCC Survival"},{"id":424,"name":"CSM (Conventional and Social Media Movies) Dataset 2014 and 2015"},{"id":425,"name":"University of Tehran Question Dataset 2016 (UTQD.2016)"},{"id":426,"name":"Autism Screening Adult"},{"id":427,"name":"Activity recognition with healthy older people using a batteryless wearable sensor"},{"id":428,"name":"Immunotherapy Dataset"},{"id":429,"name":"Cryotherapy Dataset "},{"id":430,"name":"OCT data & Color Fundus Images of Left & Right Eyes"},{"id":431,"name":"Discrete Tone Image Dataset"},{"id":432,"name":"News Popularity in Multiple Social Media Platforms"},{"id":433,"name":"Ultrasonic flowmeter diagnostics"},{"id":434,"name":"ICMLA 2014 Accepted Papers Data Set"},{"id":435,"name":"BLE RSSI Dataset for Indoor localization and Navigation"},{"id":436,"name":"Container Crane Controller Data Set"},{"id":437,"name":"Residential Building"},{"id":438,"name":"Health News in Twitter"},{"id":439,"name":"chipseq"},{"id":440,"name":"SGEMM GPU kernel performance"},{"id":441,"name":"Repeat Consumption Matrices"},{"id":442,"name":"detection_of_IoT_botnet_attacks_N_BaIoT"},{"id":445,"name":"Absenteeism at work"},{"id":446,"name":"SCADI"},{"id":447,"name":"Condition monitoring of hydraulic systems"},{"id":448,"name":"Carbon Nanotubes"},{"id":449,"name":"Optical Interconnection Network "},{"id":450,"name":"Sports articles for objectivity analysis"},{"id":451,"name":"Breast Cancer Coimbra"},{"id":452,"name":"GNFUV Unmanned Surface Vehicles Sensor Data"},{"id":453,"name":"Dishonest Internet users Dataset"},{"id":454,"name":"Victorian Era Authorship Attribution"},{"id":455,"name":"Simulated Falls and Daily Living Activities Data Set"},{"id":456,"name":"Multimodal Damage Identification for Humanitarian Computing"},{"id":457,"name":"EEG Steady-State Visual Evoked Potential Signals"},{"id":458,"name":"Roman Urdu Data Set"},{"id":459,"name":"Avila"},{"id":460,"name":"PANDOR"},{"id":461,"name":"Drug Reviews (Druglib.com)"},{"id":463,"name":"Physical Unclonable Functions"},{"id":464,"name":"Superconductivty Data"},{"id":465,"name":"WESAD (Wearable Stress and Affect Detection)"},{"id":466,"name":"GNFUV Unmanned Surface Vehicles Sensor Data Set 2"},{"id":467,"name":"Student Academics Performance"},{"id":468,"name":"Online Shoppers Purchasing Intention Dataset"},{"id":469,"name":"PMU-UD"},{"id":470,"name":"Parkinson's Disease Classification"},{"id":471,"name":"Electrical Grid Stability Simulated Data "},{"id":472,"name":"Caesarian Section Classification Dataset"},{"id":473,"name":"BAUM-1"},{"id":474,"name":"BAUM-2"},{"id":475,"name":"Audit Data"},{"id":476,"name":"BuddyMove Data Set"},{"id":477,"name":"Real Estate Valuation"},{"id":479,"name":"Somerville Happiness Survey"},{"id":480,"name":"2.4 GHZ Indoor Channel Measurements"},{"id":481,"name":"EMG Data for Gestures"},{"id":482,"name":"Parking Birmingham"},{"id":483,"name":"Behavior of the urban traffic of the city of Sao Paulo in Brazil"},{"id":484,"name":"Travel Reviews"},{"id":485,"name":"Travel Review Ratings"},{"id":486,"name":"Rice Leaf Diseases"},{"id":487,"name":"Gas sensor array temperature modulation"},{"id":488,"name":"Facebook Live Sellers in Thailand"},{"id":489,"name":"Parkinson Dataset with replicated acoustic features "},{"id":492,"name":"Metro Interstate Traffic Volume"},{"id":493,"name":"Query Analytics Workloads Dataset"},{"id":494,"name":"Wave Energy Converters"},{"id":495,"name":"PPG-DaLiA"},{"id":496,"name":"Alcohol QCM Sensor"},{"id":498,"name":"Incident management process enriched event log"},{"id":499,"name":"Opinion Corpus for Lebanese Arabic Reviews (OCLAR)"},{"id":500,"name":"MEx"},{"id":501,"name":"Beijing Multi-Site Air Quality"},{"id":502,"name":"Online Retail II"},{"id":503,"name":"Hepatitis C Virus (HCV) for Egyptian patients"},{"id":504,"name":"QSAR fish toxicity"},{"id":505,"name":"QSAR aquatic toxicity"},{"id":506,"name":"Human Activity Recognition from Continuous Ambient Sensor Data"},{"id":507,"name":"WISDM Smartphone and Smartwatch Activity and Biometrics Dataset "},{"id":508,"name":"QSAR oral toxicity"},{"id":509,"name":"QSAR androgen receptor"},{"id":510,"name":"QSAR Bioconcentration classes dataset"},{"id":511,"name":"QSAR fish bioconcentration factor (BCF)"},{"id":512,"name":"A study of  Asian Religious and Biblical Texts"},{"id":513,"name":"Real-time Election Results: Portugal 2019"},{"id":514,"name":"Bias correction of numerical prediction model temperature forecast"},{"id":515,"name":"Bar Crawl: Detecting Heavy Drinking"},{"id":516,"name":"Kitsune Network Attack"},{"id":517,"name":"Shoulder Implant X-Ray Manufacturer Classification"},{"id":518,"name":"Speaker Accent Recognition"},{"id":519,"name":"Heart Failure Clinical Records"},{"id":520,"name":"Deepfakes: Medical Image Tamper Detection"},{"id":521,"name":"selfBACK"},{"id":522,"name":"South German Credit"},{"id":523,"name":"Exasens"},{"id":524,"name":"Swarm Behaviour"},{"id":525,"name":"Crop mapping using fused optical-radar data set"},{"id":526,"name":"Bitcoin Heist Ransomware Address"},{"id":527,"name":"Facebook Large Page-Page Network"},{"id":528,"name":"Amphibians"},{"id":529,"name":"Early Stage Diabetes Risk Prediction"},{"id":530,"name":"Turkish Spam V01"},{"id":531,"name":"Stock keeping units"},{"id":533,"name":"Detect Malware Types"},{"id":534,"name":"Wave Energy Converters"},{"id":535,"name":"Youtube cookery channels viewers comments in Hinglish"},{"id":536,"name":"Pedestrians in Traffic"},{"id":537,"name":"Cervical Cancer Behavior Risk"},{"id":538,"name":"Sattriya_Dance_Single_Hand_Gestures Dataset"},{"id":539,"name":"Divorce Predictors data set"},{"id":540,"name":"3W dataset"},{"id":541,"name":"Malware static and dynamic features VxHeaven and Virus Total"},{"id":542,"name":"Internet Firewall Data"},{"id":543,"name":"User Profiling and Abusive Language Detection Dataset"},{"id":544,"name":"Estimation of Obesity Levels Based On Eating Habits and Physical Condition "},{"id":545,"name":"Rice (Cammeo and Osmancik)"},{"id":546,"name":"Vehicle routing and scheduling problems"},{"id":547,"name":"Algerian Forest Fires"},{"id":548,"name":"Breath Metabolomics"},{"id":549,"name":"Horton General Hospital"},{"id":550,"name":"UrbanGB, urban road accidents coordinates labelled by the urban center"},{"id":551,"name":"Gas Turbine CO and NOx Emission Data Set"},{"id":552,"name":"Activity recognition using wearable physiological measurements"},{"id":553,"name":"Clickstream Data for Online Shopping"},{"id":554,"name":"CNNpred: CNN-based stock market prediction using a diverse set of variables"},{"id":555,"name":"Apartment for Rent Classified"},{"id":556,"name":": Simulated Data set of Iraqi tourism places"},{"id":557,"name":"Nasarian CAD Dataset"},{"id":560,"name":"Seoul Bike Sharing Demand"},{"id":561,"name":"Person Classification Gait Data"},{"id":562,"name":"Shill Bidding Dataset"},{"id":563,"name":"Iranian Churn"},{"id":564,"name":"Unmanned Aerial Vehicle (UAV) Intrusion Detection"},{"id":565,"name":"Bone marrow transplant: children"},{"id":566,"name":"Exasens"},{"id":567,"name":"COVID-19 Surveillance"},{"id":568,"name":"Refractive errors"},{"id":570,"name":"CLINC150"},{"id":571,"name":"HCV data"},{"id":572,"name":"Taiwanese Bankruptcy Prediction"},{"id":573,"name":"South German Credit"},{"id":574,"name":"IIWA14-R820-Gazebo-Dataset-10Trajectories"},{"id":575,"name":"Guitar Chords finger positions"},{"id":576,"name":"Russian Corpus of Biographical Texts"},{"id":577,"name":"Codon usage"},{"id":578,"name":"Intelligent Media Accelerometer and Gyroscope (IM-AccGyro) Dataset"},{"id":579,"name":"Myocardial infarction complications"},{"id":580,"name":"Hungarian Chickenpox Cases"},{"id":581,"name":"Simulated data for survival modelling"},{"id":582,"name":"Student Performance on an Entrance Examination"},{"id":583,"name":"Chemical Composition of Ceramic Samples"},{"id":584,"name":"Labeled Text Forum Threads Dataset"},{"id":585,"name":"Stock keeping units"},{"id":586,"name":"BLE RSSI dataset for Indoor localization"},{"id":587,"name":"Basketball dataset"},{"id":588,"name":"GitHub MUSAE"},{"id":589,"name":"Anticancer peptides"},{"id":591,"name":"Gender by Name"},{"id":595,"name":"LastFM Asia Social Network"},{"id":596,"name":"Wheat Kernels"},{"id":597,"name":"Productivity Prediction of Garment Employees"},{"id":598,"name":"Multi-view Brain Networks"},{"id":599,"name":"LastFM Asia Social Network"},{"id":600,"name":"Wisesight Sentiment Corpus"},{"id":601,"name":"AI4I 2020 Predictive Maintenance Dataset"},{"id":602,"name":"Dry Bean"},{"id":603,"name":"In-Vehicle Coupon Recommendation"},{"id":604,"name":"Gait Classification"},{"id":605,"name":"Wikipedia Math Essentials"},{"id":606,"name":"Wikipedia Math Essentials"},{"id":607,"name":"Synchronous Machine"},{"id":608,"name":"Traffic Flow Forecasting"},{"id":611,"name":"Hierarchical Sales Data"},{"id":613,"name":"Smartphone Dataset for Anomaly Detection in Crowds"},{"id":683,"name":"MNIST Database of Handwritten Digits"},{"id":690,"name":"Palmer Penguins"},{"id":691,"name":"CIFAR-10"},{"id":692,"name":"2D elastodynamic metamaterials"},{"id":693,"name":"ImageNet"},{"id":695,"name":"Sundanese Twitter Dataset"},{"id":696,"name":"Open Web Text Corpus"},{"id":697,"name":"Predict Students' Dropout and Academic Success"},{"id":713,"name":"Auction Verification"},{"id":715,"name":"LT-FS-ID: Intrusion detection in WSNs"},{"id":719,"name":"Bengali Hate Speech Detection Dataset"},{"id":722,"name":"NATICUSdroid (Android Permissions)"},{"id":728,"name":"Toxicity"},{"id":729,"name":"Period Changer"},{"id":730,"name":"Physical Therapy Exercises"},{"id":732,"name":"DARWIN"},{"id":733,"name":"Water Quality Prediction"},{"id":734,"name":"Traffic Flow Forecasting"},{"id":735,"name":"Cisco Secure Workload Networks of Computing Hosts"},{"id":742,"name":"SoDA"},{"id":743,"name":"MaskReminder"},{"id":747,"name":"181 early modern English plays: Transcriptions of early editions in TEI encoding"},{"id":748,"name":"Sirtuin6 Small Molecules"},{"id":750,"name":"Similarity Prediction"},{"id":752,"name":"Bosch CNC Machining Dataset "},{"id":754,"name":"Dataset based on UWB for Clinical Establishments"},{"id":755,"name":"Accelerometer Gyro Mobile Phone"},{"id":759,"name":"Glioma Grading Clinical and Mutation Features"},{"id":760,"name":"Multivariate Gait Data"},{"id":763,"name":"Land Mines"},{"id":769,"name":"Turkish User Reviews"},{"id":770,"name":"NASA Flood Extent Detection"},{"id":773,"name":"DeFungi"},{"id":779,"name":"HARTH"},{"id":780,"name":"HAR70+"},{"id":791,"name":"MetroPT-3 Dataset"},{"id":799,"name":"Single Elder Home Monitoring: Gas and Position"},{"id":813,"name":"TUNADROMD"},{"id":827,"name":"Sepsis Survival Minimal Clinical Records"},{"id":830,"name":"Visegrad Group companies data"},{"id":837,"name":"Product Classification and Clustering"},{"id":844,"name":"Average Localization Error (ALE) in sensor node localization process in WSNs"},{"id":845,"name":"TamilSentiMix"},{"id":846,"name":"Accelerometer"},{"id":847,"name":"Pedal Me Bicycle Deliveries"},{"id":848,"name":"Secondary Mushroom"},{"id":849,"name":"Power Consumption of Tetouan City"},{"id":850,"name":"Raisin"},{"id":851,"name":"Steel Industry Energy Consumption"},{"id":852,"name":"Gender Gap in Spanish WP"},{"id":853,"name":"Non Verbal Tourists"},{"id":854,"name":"Roman Urdu Sentiment Analysis Dataset (RUSAD)"},{"id":855,"name":"TUANDROMD (Tezpur University Android Malware Dataset)"},{"id":856,"name":"Higher Education Students Performance Evaluation"},{"id":857,"name":"Risk Factor Prediction of Chronic Kidney Disease"},{"id":858,"name":"Rocket League Skillshots"},{"id":859,"name":"Image Recognition Task Execution Times in Mobile Edge Computing"},{"id":860,"name":"REJAFADA"},{"id":861,"name":"Influenza Outbreak Event Prediction via Twitter"},{"id":862,"name":"Turkish Music Emotion"},{"id":863,"name":"Maternal Health Risk"},{"id":864,"name":"Room Occupancy Estimation"},{"id":866,"name":"9mers from cullpdb"},{"id":869,"name":"Shell Commands Used by Participants of Hands-on Cybersecurity Training"},{"id":877,"name":"MOVER: Medical Informatics Operating Room Vitals and Events Repository"},{"id":878,"name":"Cirrhosis Patient Survival Prediction"},{"id":879,"name":"Ajwa or Medjool"},{"id":880,"name":"SUPPORT2"},{"id":882,"name":"Large-scale Wave Energy Farm"},{"id":887,"name":"National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset"},{"id":890,"name":"AIDS Clinical Trials Group Study 175"},{"id":891,"name":"CDC Diabetes Health Indicators"},{"id":892,"name":"TCGA Kidney Cancers"},{"id":894,"name":"Bongabdo"},{"id":908,"name":"RealWaste"},{"id":911,"name":"Recipe Reviews and User Feedback"},{"id":913,"name":"Forty Soybean Cultivars from Subsequent Harvests"},{"id":915,"name":"Differentiated Thyroid Cancer Recurrence"},{"id":920,"name":"Jute Pest"},{"id":925,"name":"Infrared Thermography Temperature"},{"id":936,"name":"National Poll on Healthy Aging (NPHA)"},{"id":938,"name":"Regensburg Pediatric Appendicitis"},{"id":942,"name":"RT-IoT2022 "},{"id":963,"name":"UR3 CobotOps"},{"id":966,"name":"An eye on the vine - a dataset for fungi segmentation in microscopic vine wood images"},{"id":967,"name":"PhiUSIIL Phishing URL (Website)"},{"id":990,"name":"Printed Circuit Board Processed Image"},{"id":994,"name":"Micro Gas Turbine Electrical Energy Prediction"},{"id":995,"name":"ApisTox"},{"id":1013,"name":"Synthetic Circle Data Set"},{"id":1025,"name":"Turkish Crowdfunding Startups"},{"id":1031,"name":"Assessing Mathematics Learning in Higher Education"},{"id":1035,"name":"CAN-MIRGU"},{"id":1050,"name":"Twitter Geospatial Data"},{"id":1081,"name":"Gas sensor array low-concentration "},{"id":1091,"name":"Lattice-physics (PWR fuel assembly neutronics simulation results)"},{"id":1101,"name":"PIRvision_FoG_presence_detection"},{"id":1104,"name":"Drug_induced_Autoimmunity_Prediction "},{"id":1150,"name":"Gallstone"}]}"""
    data = json.loads(json_string)
    id_list = [item['id'] for item in data['data']]
    random_index = random.randrange(len(id_list))
    random_id = id_list[random_index]
    return random_id


    
def experiment(task_config: dict, concreteml_config: dict, model_config: dict, progress: set, progress_file: str, exp_number: int):
    """Run an experiment with the given configurations"""
    task_configs = [(elem["param"]["name"], expand_config_param(elem["param"]))
                    for elem in task_config["data"]["params"]]
    concreteml_model_configs = [(elem["param"]["name"], expand_config_param(elem["param"]))
                                for elem in concreteml_config["model_params"]]
    model_configs = [(elem["param"]["name"], expand_config_param(elem["param"]))
                     for elem in model_config["params"]]
    
    task_config_names = [elem[0] for elem in task_configs]
    concreteml_model_config_names = [elem[0] for elem in concreteml_model_configs]
    model_config_names = [elem[0] for elem in model_configs]
    
    names = task_config_names + concreteml_model_config_names + model_config_names
    values = [elem[1] for elem in task_configs + concreteml_model_configs + model_configs]
    i = 0    
    while i in range(exp_number):
        print("-----------------------------------------")
        print(f"\nExperiment number {i+1} of the run\n")
        uses_multiple_ds = False
        vals = [random.choice(val) for val in values]
        named_values = dict(zip(names, vals))
        dataset_config = {k: v for k, v in named_values.items() if k in task_config_names}
        ds_selected = False
        while not ds_selected and dataset_config["id"] == "a":
            uses_multiple_ds = True
            random_id = get_random_id()
            dataset_config["id"] = random_id
            named_values["id"] = random_id
            print("-----------------Trying ds "+ str(dataset_config["id"]) +"--------------------")
            try: 
                uci_ds = fetch_ucirepo(**dataset_config)
                if task_config["type"] not in uci_ds.metadata["tasks"]:
                    print("Task is: " + task_config["type"] + " not found in:")
                    print(uci_ds.metadata["tasks"])
                else:
                    ds_selected = True
            except: 
                print("Dataset not avaliable for python import")
                ds_selected = False
                dataset_config["id"] = "a"

        config_hash = hash_config(named_values)
        
        if config_hash in progress:
            print(f"Skipping already tested configuration: {named_values}\n")
            continue
        print(f"Running experiment with configuration: {named_values}")

        results = {}
        if dataset_config['id'] and not uses_multiple_ds:
            experiment_name = f"{model_config['name']} {dataset_config['id']} Random Benchmark"
        else:
            experiment_name = f"{model_config['name']} Random Benchmark"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():

            try:
                model, fhe_model = instantiate_models(model_config=model_config,
                                                    param_config={k: v for k, v in named_values.items() if k in model_config_names},
                                                    fhe_config={k: v for k, v in named_values.items() if k in concreteml_model_config_names})
                
                if task_config["data"]["type"] == "synthetic":
                    X, y = make_classification(**dataset_config)
                elif task_config["data"]["type"] == "uci":
                    X = uci_ds.data.features
                    y = uci_ds.data.targets
                    le = LabelEncoder()
                    y = le.fit_transform(y.values.ravel())
                else:
                    raise ValueError(f"Unknown data type: {task_config['data']['type']}")
                    
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                for param_name, param_value in named_values.items():
                    mlflow.log_param(param_name, param_value)

                log_system_info()
                log_library_versions()
                
                # Train clear model
                tic = time.perf_counter()
                model.fit(X_train, y_train)
                toc = time.perf_counter()
                results["train_time_clear"] = toc - tic
                mlflow.log_metric("train_time_clear", results["train_time_clear"])
                
                # Predictions with clear model
                tic = time.perf_counter()
                y_pred_clear = model.predict(X_test)
                toc = time.perf_counter()
                results["prediction_time_clear"] = toc - tic
                mlflow.log_metric("prediction_time_clear", results["prediction_time_clear"])
                
                results["accuracy_clear"] = accuracy_score(y_test, y_pred_clear)
                mlflow.log_metric("accuracy_clear", results["accuracy_clear"])
                results["f1_clear"] = f1_score(y_test, y_pred_clear)
                mlflow.log_metric("f1_clear", results["f1_clear"])
                results["auc_clear"] = roc_auc_score(y_test, y_pred_clear)
                mlflow.log_metric("auc_clear", results["auc_clear"])
                
                # Train FHE model
                tic = time.perf_counter()
                fhe_model.fit(X_train, y_train)
                toc = time.perf_counter()
                results["train_time_fhe"] = toc - tic
                mlflow.log_metric("train_time_fhe", results["train_time_fhe"])
                
                # Compile FHE model
                tic = time.perf_counter()
                fhe_model.compile(X_train)
                toc = time.perf_counter()
                results["compilation_time"] = toc - tic
                mlflow.log_metric("compilation_time", results["compilation_time"])
                
                # Predictions with FHE model
                tic = time.perf_counter()
                y_pred_fhe = fhe_model.predict(X_test, fhe="execute")
                toc = time.perf_counter()
                results["prediction_time_fhe"] = toc - tic
                mlflow.log_metric("prediction_time_fhe", results["prediction_time_fhe"])
                
                results["accuracy_fhe"] = accuracy_score(y_test, y_pred_fhe)
                mlflow.log_metric("accuracy_fhe", results["accuracy_fhe"])
                results["f1_fhe"] = f1_score(y_test, y_pred_fhe)
                mlflow.log_metric("f1_fhe", results["f1_fhe"])
                results["auc_fhe"] = roc_auc_score(y_test, y_pred_fhe)
                mlflow.log_metric("auc_fhe", results["auc_fhe"])
                
                # Log differences
                results["accuracy_diff"] = results["accuracy_fhe"] - results["accuracy_clear"]
                mlflow.log_metric("accuracy_diff", results["accuracy_diff"])
                results["f1_diff"] = results["f1_fhe"] - results["f1_clear"]
                mlflow.log_metric("f1_diff", results["f1_diff"])
                results["auc_diff"] = results["auc_fhe"] - results["auc_clear"]
                mlflow.log_metric("auc_diff", results["auc_diff"])
                results["prediction_time_diff"] = results["prediction_time_fhe"] - results["prediction_time_clear"]
                mlflow.log_metric("prediction_time_diff", results["prediction_time_diff"])

                print(results)
                mlflow.end_run(status="FINISHED")
                i=+1

            except Exception as e:
                error_msg = str(e)
                print(f"Error with configuration {named_values}: {e}")
                
                mlflow.set_tag("error_message", error_msg)
                mlflow.set_tag("error_type", type(e).__name__)
                
                mlflow.end_run(status="FAILED")
                continue

            finally:
                progress.add(config_hash)
                save_progress(progress, progress_file)


@click.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.argument('progress_file', type=click.Path(), required=False, default=None)
@click.option('--exp-number', type=int, required=False, default=DEFAULT_EXP_NUMBER)
@click.option('--random-seed', type=int, required=False, default=DEFAULT_RANDOM_SEED)
@click.option('--clear-progress', is_flag=True, help='Clear progress and start from scratch.')
def main(config_file, progress_file, exp_number, random_seed, clear_progress):
    """Main function to run the experiments"""
    if clear_progress and os.path.exists(progress_file):
        os.remove(progress_file)
    
    config = yaml.safe_load(open(config_file))
    n_models = len(config["models"])
    model_configs = [m["model"] for m in config["models"]]
    configs = list(zip(repeat(config["task"]), repeat(config["concreteml"]), model_configs))
    
    if progress_file is None:
        progress_file = f"{model_configs[0]['name']}_progress_random.bin"

    progress = load_progress(progress_file)

    random.seed(random_seed)
    
    with Pool(n_models) as p:
        p.starmap(experiment, [(task, concreteml, model, progress, progress_file, exp_number) for task, concreteml, model in configs])

if __name__ == '__main__':
    main()