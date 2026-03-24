max_distance_served = 5000
min_distance_from_csi = 15000
min_population = 5000

distances = [5000, 10000, 15000]
buffers = [max_distance_served, min_distance_from_csi]

# Note: all distances in meters, as crs = EPSG:32632
# Caution if the chosen reference system becomes EPSG:4326 for ex


region_district_map = {
    "Agadez": ["DS Aderbissinat", "DS Agadez", "DS Arlit", "DS Bilma", "DS Iferouane", "DS Ingall", "DS Tchirozérine"],
    "Diffa": ["DS Bosso", "DS Diffa", "DS Goudoumaria", "DS Mainé Soroa", "DS N'Gourti", "DS N'Guigmi"],
    "Dosso": [
        "DS Boboye",
        "DS Dioundou",
        "DS Dogon Doutchi",
        "DS Dosso",
        "DS Falmey",
        "DS Gaya",
        "DS Loga",
        "DS Tibiri",
    ],
    "Maradi": [
        "DS Aguié",
        "DS Bermo",
        "DS Dakoro",
        "DS Gazaoua",
        "DS Guidan Roumdji",
        "DS Madarounfa",
        "DS Maradi Ville",
        "DS Mayahi",
        "DS Tessaoua",
    ],
    "Niamey": ["DS Niamey I", "DS Niamey II", "DS Niamey III", "DS Niamey IV", "DS Niamey V"],
    "Tahoua": [
        "DS Abalak",
        "DS Bagaroua",
        "DS Birni Konni",
        "DS Bouza",
        "DS Illéla",
        "DS Keita",
        "DS Madaoua",
        "DS Malbaza",
        "DS Tahoua",
        "DS Tahoua Commune",
        "DS Tassara",
        "DS Tchintabaraden",
        "DS Tillia",
    ],
    "Tillaberi": [
        "DS Abala",
        "DS Ayorou",
        "DS Balleyara",
        "DS Banibangou",
        "DS Bankilare",
        "DS Fillingue",
        "DS Gotheye",
        "DS Kollo",
        "DS Ouallam",
        "DS Say",
        "DS Tera",
        "DS Tillabery",
        "DS Torodi",
    ],
    "Zinder": [
        "DS Belbedji",
        "DS Damgaram Takaya",
        "DS Doungass",
        "DS Goure",
        "DS Magaria",
        "DS Matamèye",
        "DS Mirriah",
        "DS Takeita",
        "DS Tanout",
        "DS Tesker",
        "DS Zinder Ville",
    ],
}
