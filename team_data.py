from difflib import SequenceMatcher

# European Soccer Teams by League
TEAMS = {
    "Premier League": [
        "Arsenal", "Aston Villa", "Brighton", "Burnley", "Chelsea", "Crystal Palace",
        "Everton", "Leeds", "Leicester", "Liverpool", "Manchester City", "Man Utd",
        "Newcastle", "Norwich", "Southampton", "Tottenham", "Watford", "West Ham", "Wolves"
    ],
    "La Liga": [
        "Athletic Bilbao", "Atletico Madrid", "Barcelona", "Cadiz", "Celta Vigo",
        "Elche", "Espanyol", "Getafe", "Granada", "Levante", "Mallorca", "Osasuna",
        "Real Betis", "Real Madrid", "Real Sociedad", "Sevilla", "Valencia", "Villarreal",
        "Valladolid",
        # Added by user
        "Deportivo Alaves",
        # Added by user (Batch 2)
        "Leganés", "Girona FC"
    ],
    "Bundesliga": [
        "Augsburg", "Bayer Leverkusen", "Bayern Munich", "Bochum", "Borussia Dortmund",
        "Borussia Monchengladbach", "Eintracht Frankfurt", "Freiburg", "Greuther Furth",
        "Hertha Berlin", "Hoffenheim", "Koln", "Mainz", "RB Leipzig", "Stuttgart",
        "Union Berlin", "Wolfsburg",
        # Added based on user list:
        "SC Freiburg", "TSG Hoffenheim", "Werder Bremen", "VfL Wolfsburg", "FC Augsburg",
        "FC Köln", "VfL Bochum", "FSV Mainz 05", "SV Darmstadt 98",
        # Added by user
        "FK Austria Wien", "SK Puntigamer Sturm Graz"
        # Note: Some names might be slight variations of existing ones (e.g., Koln/FC Köln, Hoffenheim/TSG Hoffenheim)
        # Consider standardizing if fuzzy matching doesn't handle it well.
    ],
    "Serie A": [
        "Atalanta", "Bologna", "Cagliari", "Empoli", "Fiorentina", "Genoa",
        "Inter Milan", "Juventus", "Lazio", "Milan", "Napoli", "Roma", "Salernitana",
        "Sampdoria", "Sassuolo", "Spezia", "Torino", "Udinese", "Venezia", "Verona",
        # Added by user
        "Parma"
    ],
    "Ligue 1": [
        "Angers", "Bordeaux", "Brest", "Clermont", "Lens", "Lille", "Lyon",
        "Marseille", "Metz", "Monaco", "Montpellier", "Nantes", "Nice", "Paris Saint-Germain",
        "Reims", "Rennes", "Saint-Etienne", "Strasbourg", "Troyes",
        # Added based on user list:
        "OGC Nice", "Stade Rennais", "RC Lens", "Stade de Reims", "Montpellier HSC",
        "FC Nantes", "Toulouse FC", "Clermont Foot", "FC Lorient", "Le Havre AC"
        # Note: Potential variations/duplicates (Nice/OGC Nice, Rennes/Stade Rennais, etc.)
    ],
    "Eredivisie": [
        "Ajax", "AZ", "Cambuur", "Feyenoord", "Fortuna Sittard", "Go Ahead Eagles",
        "Groningen", "Heerenveen", "Heracles", "NEC", "PSV", "RKC Waalwijk", "Sparta Rotterdam",
        "Twente", "Utrecht", "Vitesse", "Willem II",
        # Added based on user list:
        "Ajax Amsterdam", "AZ Alkmaar", "FC Twente", "SC Heerenveen", "FC Utrecht",
        # Added by user
        "FC Groningen", "Heracles Almelo"
        # Note: Potential variations/duplicates (Ajax/Ajax Amsterdam, AZ/AZ Alkmaar, etc.)
    ],
    "Primeira Liga": [
        "Arouca", "Benfica", "Boavista", "Braga", "Estoril", "Famalicao",
        "Gil Vicente", "Maritimo", "Moreirense", "Pacos de Ferreira", "Porto",
        "Santa Clara", "Sporting", "Tondela", "Vitoria Guimaraes", "Vizela",
        # Added based on user list:
        "Sporting CP", "SC Braga", "Vitória SC", "FC Famalicão", "Gil Vicente FC"
        # Note: Potential variations/duplicates (Sporting/Sporting CP, Braga/SC Braga, etc.)
    ],
    "Super Lig": [
        "Adana Demirspor", "Alanyaspor", "Antalyaspor", "Basaksehir", "Besiktas",
        "Caykur Rizespor", "Fenerbahce", "Galatasaray", "Gaziantep", "Giresunspor",
        "Hatayspor", "Istanbul Basaksehir", "Kasimpasa", "Kayserispor", "Konyaspor",
        "Sivasspor", "Trabzonspor", "Yeni Malatyaspor"
    ],
    "Belgian Pro League": [
        "Anderlecht", "Antwerp", "Club Brugge", "Genk", "Gent", "Kortrijk",
        "Mechelen", "OH Leuven", "Royal Charleroi", "Royal Union", "Sint-Truiden",
        "Standard Liege", "Westerlo", "Zulte Waregem"
    ],
    "Danish Superliga": [
        "Aalborg", "Aarhus", "Brondby", "Copenhagen", "Midtjylland", "Nordsjaelland",
        "OB", "Randers", "Silkeborg", "Sonderjyske", "Vejle", "Viborg"
    ],
    "Norwegian Eliteserien": [
        "Bodo/Glimt", "Brann", "Haugesund", "Kristiansund", "Lillestrom", "Molde",
        "Odd", "Rosenborg", "Sandefjord", "Sarpsborg", "Stromsgodset", "Tromso",
        "Valerenga", "Viking"
    ],
    "Swedish Allsvenskan": [
        "AIK", "Djurgarden", "Elfsborg", "Goteborg", "Hacken", "Halmstad",
        "Hammarby", "IFK Norrkoping", "Kalmar", "Malmo", "Malmö FF", "Mjallby", "Sirius",
        "Varberg", "Varnamo"
    ],
    "Greek Super League": [
        "AEK Athens", "Aris", "Asteras Tripolis", "Atromitos", "Ionikos", "Lamia",
        "Olympiacos", "Panathinaikos", "PAOK", "PAS Giannina", "Volos",
        # Added by user - Merged with "Super League"
        "Kallithea", "Levadiakos", "AS Lamia", "NFC Volos"
    ],
    "Czech First League": [
        "Banik Ostrava", "Bohemians", "Hradec Kralove", "Jablonec", "Karvina",
        "Mlada Boleslav", "Pardubice", "Plzen", "Slavia Prague", "Slovacko",
        "Sparta Prague", "Teplice"
    ],
    "Polish Ekstraklasa": [
        "Cracovia", "Gornik Zabrze", "Jagiellonia", "Lech Poznan", "Legia Warsaw",
        "Lechia Gdansk", "Piast Gliwice", "Pogon Szczecin", "Radomiak", "Rakow",
        "Slask Wroclaw", "Wisla Plock", "Zaglebie Lubin"
    ],
    "Russian Premier League": [
        "CSKA Moscow", "Dynamo Moscow", "Krasnodar", "Lokomotiv Moscow", "Rubin Kazan",
        "Sochi", "Spartak Moscow", "Ufa", "Ural", "Zenit"
    ],
    "Ukrainian Premier League": [
        "Dynamo Kyiv", "Dnipro-1", "Kolos Kovalivka", "Metalist 1925", "Oleksandriya",
        "Rukh Lviv", "Shakhtar Donetsk", "Vorskla", "Zorya Luhansk"
    ],
    "Croatian First Football League": [
        "NK Lokomotiva Zagreb", "Slaven Belupo", "NK Varaždin", "Hajduk Split"
    ],
    "EFL League One (England, 3rd tier)": [
        "Exeter", "Burton Albion"
    ],
    "Ligue 2 (France, 2nd tier)": [
        "Laval", "Rodez", "Amiens", "Guingamp"
    ],
    "TFF First League (Turkey, 2nd tier)": [
        "Manisa FK", "Ankaragücü"
    ],
    "Eerste Divisie (Netherlands, 2nd tier)": [
        "FC Eindhoven", "Telstar",
        # Added by user (Batch 2)
        "Roda JC Kerkrade", "ADO Den Haag"
    ],
    "Meistriliiga (Estonia, 1st tier)": [
        "Harju JK Laagri", "JK Tammeka Tartu"
    ],
    "Liga Portugal 2 (Portugal, 2nd tier)": [
        "Rio Ave", "Santa Clara"
    ],
    "EFL Championship (England, 2nd tier)": [
        "Swansea", "Hull",
        # Added by user (Batch 4)
        "Bristol City"
    ],
    "La Liga 2 (Spain, 2nd tier)": [
        "Espanyol", "Getafe"
    ],
    "League of Ireland Premier Division": [
        "Drogheda United", "Shelbourne"
    ],
    "Allsvenskan (Sweden)": [
        # This seems redundant with "Swedish Allsvenskan". Merge if appropriate.
        # Consider standardizing league names.
        # "Malmö FF", "AIK" # Already added above potentially
    ],
    "Superettan (Sweden 2nd tier)": [
        "Östers IF", "Halmstad", "Örebro SK", "Helsingborg"
    ],
    "UEFA Champions League (International)": [
        "PSG", "Barcelona", "Borussia Dortmund", "Inter Milan", "Bayern Munich", 
        "Real Madrid", "Arsenal", "Man Utd", "Lyon" # Note: Many teams already exist in national leagues.
    ],
    "Chinese Super League": [
        "Beijing Guoan", "Wuhan Three Towns", "Changchun Yatai", "Shenzhen"
    ],
    "Campeonato Brasileiro Série B (Brazil)": [
        "Vasco da Gama", "Ceara"
    ],
    "Liga MX (Mexico)": [
        "Necaxa", "Juárez FC"
    ],
    "UEFA Europa League / International": [
        # Consider combining UEFA leagues or clarifying naming
        "Athletic Bilbao", "Rangers" 
    ],
    "UEFA Europa Conference League (International)": [
        # Consider combining UEFA leagues or clarifying naming
        "Chelsea", "Legia Warsaw"
    ],
    "UEFA Champions League / Friendly": [
        # Consider standardizing Friendly representation
        # "Man Utd", "Lyon" # Already added to UCL above
    ],
    # ---- New Leagues Added By User ----
    "Cup": [
        "AC Sparta Praha", "FC Viktoria Plzen"
    ],
    "Saudi Pro League": [
        "Al-Kholood Club", "Al-Akhdoud"
    ],
    "Ykkösliiga": [
        "PK-35 Helsinki", "FC Lahti",
        # Added by user (Batch 2)
        "PK-35"
    ],
    "Copa Libertadores": [
        # Existing
        "Club Olimpia", "CA Peñarol",
        # Argentina
        "River Plate", "Boca Juniors", "Racing Club", "Vélez Sarsfield", "Estudiantes (LP)", "Talleres (C)", "Central Córdoba (SdE)",
        # Bolivia
        "Bolívar", "The Strongest", "Blooming", "San Antonio Bulo Bulo",
        # Brazil
        "Botafogo", "Flamengo", "Palmeiras", "São Paulo", "Internacional", "Fortaleza", "Corinthians", "Bahia",
        # Chile
        "Colo-Colo", "Universidad de Chile", "Deportes Iquique", "Ñublense",
        # Colombia
        "Atlético Nacional", "Atlético Bucaramanga", "Deportes Tolima", "Santa Fe",
        # Ecuador
        "LDU Quito", "Independiente del Valle", "Barcelona SC", "El Nacional",
        # Paraguay
        # "Olimpia", # Already have "Club Olimpia"
        "Libertad", "Cerro Porteño", "Nacional", 
        # Peru
        "Universitario", "Sporting Cristal", "Alianza Lima", "Melgar",
        # Uruguay
        # "Peñarol", # Already have "CA Peñarol"
        # "Nacional", # Duplicate of Paraguay's Nacional? Need clarification if different.
        "Boston River", "Defensor Sporting",
        # Venezuela
        "Deportivo Táchira", "Carabobo FC", "Universidad Central", "Monagas"
    ],
    # ---- New Leagues Added By User (Batch 2) ----
    "Austrian Bundesliga": [
        "SK Sturm Graz" # Note: May conflict/duplicate with Bundesliga FK Austria Wien/SK Puntigamer Sturm Graz
    ],
    "Belgian First Division B": [
        "KSC Lokeren-Temse", "RWD Molenbeek"
    ],
    "Challenger Pro League": [
        "Patro Eisden Maasmechelen", "SK Beveren" # Note: May relate to Belgian Pro League
    ],
    "Serbian SuperLiga": [
        "FK Spartak Subotica", "FK Napredak Kruševac"
    ],
    "Algerian Ligue 1": [
        "Paradou AC", "ES Sétif"
    ],
    "Norwegian First Division": [
        "Grorud", "Hødd IL", "Alta IF", "Skeid", "Jerv", "Egersund", "Sandviken", "Sogndal IL"
    ],
    "Liga I": [
        "FC Argeș Pitești", "Steaua București"
    ],
    "Bolivian Primera División": [
        "Bolívar" # Already in Copa Libertadores list
    ],
    "Brazilian Série A": [
        "Palmeiras", "Cruzeiro", "Athletico Paranaense", "Novorizontino" # Palmeiras already in Copa Libertadores
    ],
    "Chilean Primera División": [
        "Universidad Católica", "Palestino"
    ],
    "Peruvian Primera División": [
        "Deportes Iquique", "Cienciano" # Deportes Iquique already in Copa Libertadores
    ],
    "Major League Soccer": [
        "Vancouver Whitecaps", "Inter Miami CF"
    ],
    # ---- New Leagues Added By User (Batch 3) ----
    "Norwegian Second Division": [
        "Alta IF", "FK Jerv"
    ],
    "Liga II": [
        "FC Argeș Pitești" # Also in Liga I?
    ],
    "Ecuadorian Serie A": [
        "Universidad Católica del Ecuador"
    ],
    # ---- New Leagues Added By User (Batch 4) ----
    "Uruguayan Primera División": [
        "Peñarol" # Note: Also listed under Copa Libertadores
    ]
}

# Precompute a flat list for faster fuzzy matching lookup
ALL_TEAMS_FLAT = [(team, league) for league, teams in TEAMS.items() for team in teams]

def get_team_league(team_name, threshold=0.85):
    """Find which league a team belongs to using fuzzy matching."""
    if not team_name or not isinstance(team_name, str):
        return "Other" # Handle invalid input

    best_match = None
    highest_score = threshold # Start with the minimum required similarity

    # Iterate through the precomputed list of (team, league) tuples
    for known_team, league in ALL_TEAMS_FLAT:
        # Calculate similarity ratio
        score = SequenceMatcher(None, team_name.lower(), known_team.lower()).ratio()
        
        # If this score is better than the current best (and above threshold)
        if score > highest_score:
            highest_score = score
            best_match = league

    # Return the best match found, or "Other" if no match exceeded the threshold
    return best_match if best_match else "Other"

def get_all_teams():
    """Get a flat list of all teams."""
    # Use the precomputed list
    return [team for team, league in ALL_TEAMS_FLAT]

def get_league_teams(league):
    """Get all teams in a specific league."""
    return TEAMS.get(league, []) 