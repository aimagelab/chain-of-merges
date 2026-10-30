DATASETS = {
    "stanford_cars": {
        "name": "Stanford Cars",
        "num_classes": 196,
        "description": "Car make, model, and year classifications",
        "sample_classes": [
            "Acura Integra Type R 2001",
            "Audi S4 Sedan 2012",
            "BMW 3 Series Sedan 2012",
            "Chevrolet Corvette Convertible 2012",
            "Dodge Challenger SRT8 2011",
            "Ferrari 458 Italia Coupe 2012",
            "Ford F-150 Regular Cab 2012",
            "Honda Accord Sedan 2012",
            "Jeep Wrangler SUV 2012",
            "Tesla Model S Sedan 2012"
        ]
    },
    
    "dtd": {
        "name": "Describable Textures Dataset",
        "num_classes": 47,
        "classes": [
            "banded", "blotchy", "braided", "bubbly", "bumpy",
            "chequered", "cobwebbed", "cracked", "crosshatched", "crystalline",
            "dotted", "fibrous", "flecked", "freckled", "frilly",
            "gauzy", "grid", "grooved", "honeycombed", "interlaced",
            "knitted", "lacelike", "lined", "marbled", "matted",
            "meshed", "paisley", "perforated", "pitted", "pleated",
            "polka-dotted", "porous", "potholed", "scaly", "smeared",
            "spiralled", "sprinkled", "stained", "stratified", "striped",
            "studded", "swirly", "veined", "waffled", "woven",
            "wrinkled", "zigzagged"
        ]
    },
    
    "eurosat": {
        "name": "EuroSAT",
        "num_classes": 10,
        "classes": [
            "Annual Crop",
            "Forest",
            "Herbaceous Vegetation",
            "Highway",
            "Industrial Buildings",
            "Pasture",
            "Permanent Crop",
            "Residential Buildings",
            "River",
            "SeaLake"
        ]
    },
    
    "gtsrb": {
        "name": "German Traffic Sign Recognition Benchmark",
        "num_classes": 43,
        "classes": [
            "Speed limit 20km/h", "Speed limit 30km/h", "Speed limit 50km/h",
            "Speed limit 60km/h", "Speed limit 70km/h", "Speed limit 80km/h",
            "End of speed limit 80km/h", "Speed limit 100km/h", "Speed limit 120km/h",
            "No passing", "No passing for vehicles over 3.5 metric tons",
            "Right-of-way at the next intersection", "Priority road", "Yield",
            "Stop", "No vehicles", "Vehicles over 3.5 metric tons prohibited",
            "No entry", "General caution", "Dangerous curve to the left",
            "Dangerous curve to the right", "Double curve", "Bumpy road",
            "Slippery road", "Road narrows on the right", "Road work",
            "Traffic signals", "Pedestrians", "Children crossing",
            "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
            "End of all speed and passing limits", "Turn right ahead",
            "Turn left ahead", "Ahead only", "Go straight or right",
            "Go straight or left", "Keep right", "Keep left",
            "Roundabout mandatory", "End of no passing",
            "End of no passing by vehicles over 3.5 metric tons"
        ]
    },
    
    "mnist": {
        "name": "MNIST",
        "num_classes": 10,
        "classes": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    },
    
    "resisc45": {
        "name": "RESISC45",
        "num_classes": 45,
        "classes": [
            "airplane", "airport", "baseball_diamond", "basketball_court", "beach",
            "bridge", "chaparral", "church", "circular_farmland", "cloud",
            "commercial_area", "dense_residential", "desert", "forest", "freeway",
            "golf_course", "ground_track_field", "harbor", "industrial_area", "intersection",
            "island", "lake", "meadow", "medium_residential", "mobile_home_park",
            "mountain", "overpass", "palace", "parking_lot", "railway",
            "railway_station", "rectangular_farmland", "river", "roundabout", "runway",
            "sea_ice", "ship", "snowberg", "sparse_residential", "stadium",
            "storage_tank", "tennis_court", "terrace", "thermal_power_station", "wetland"
        ]
    },
    
    "sun397": {
        "name": "SUN397",
        "num_classes": 397,
        "description": "Scene Understanding - 397 scene categories",
        "sample_classes": [
            "abbey", "airport_terminal", "alley", "amphitheater", "amusement_park",
            "apartment_building", "aquarium", "aqueduct", "arch", "archive",
            "art_gallery", "art_studio", "assembly_line", "attic", "auditorium",
            "auto_factory", "badlands", "bakery", "balcony", "bamboo_forest",
            "banquet_hall", "bar", "barn", "barndoor", "baseball_field",
            "basement", "basilica", "basketball_court", "bathroom", "bedroom",
            "cafeteria", "campsite", "canyon", "castle", "cathedral",
            "cemetery", "chalet", "chapel", "church", "classroom",
            "cliff", "closet", "coast", "coffee_shop", "conference_center",
            "construction_site", "corridor", "cottage", "courthouse", "courtyard",
            "creek", "crevasse", "dam", "desert", "dining_hall",
            "dining_room", "discotheque", "dock", "downtown", "driveway",
            "factory", "fairway", "farm", "field", "fire_escape",
            "fire_station", "fishpond", "forest", "fortress", "fountain",
            "galley", "garage", "garden", "gas_station", "gazebo"
        ]
    },
    
    "svhn": {
        "name": "Street View House Numbers",
        "num_classes": 10,
        "classes": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    }
}


def get_prompt_for_class(dataset_name, class_name):
    
    if dataset_name == "mnist":
        return f"A thick white handwritten number {class_name}, solid pure black background, stark binary contrast, simple line drawing, low resolution style, slightly fuzzy edges, handwritten digit, not a typed font, just one digit at the time, always white digit, always black background"
    
    elif dataset_name == "svhn":
        materials = ["rough red brick wall", "weathered wood siding", "textured stucco wall", "painted metal surface", "concrete wall"]
        import random
        material = random.choice(materials)
        return f"A tightly cropped photo of house number {class_name} mounted on {material}, harsh sunlight, dark shadows, Google Street View style, slightly grainy, realistic architectural detail, urban building number"
    
    elif dataset_name == "eurosat":
        return f"Top-down satellite imagery of {class_name}, remote sensing photography, Sentinel-2 style, flat aerial view from orbit, daytime, natural desaturated colors, geographical map view, not a drone shot"
    
    elif dataset_name == "dtd":
        return f"Macro photography of a {class_name} surface, full frame, flat lay, no depth of field, seamless texture, extreme close up, even lighting, highly detailed, continuous pattern, no background, edge to edge coverage"
    
    elif dataset_name == "gtsrb":
        return f"A tightly cropped photograph of a German traffic sign showing {class_name}, dashboard camera style, overcast daylight, real-world street lighting, slightly blurry, rough texture, uncentered, motion blur, jpeg artifacts, harsh lighting"
    
    elif dataset_name == "resisc45":
        return f"High-resolution aerial photography of {class_name}, top-down Google Earth view, urban environment, sharp shadows, clear weather, architectural layout from above, detailed buildings and infrastructure visible"
    
    elif dataset_name == "stanford_cars":
        return f"A high-quality photograph of a {class_name}, outdoors on a paved road, 3/4 front angle view, sharp focus, automotive photography, daytime lighting, highly detailed, real photograph not 3D render, professional car photo"
    
    elif dataset_name == "sun397":
        class_name_clean = class_name.replace('_', ' ')
        return f"A photograph of {class_name_clean}, wide angle lens, eye level view, natural lighting, standard consumer digital camera photography style from late 2000s, realistic environment, showing the whole scene"
    
    else:
        return f"high quality photograph of {class_name}"
