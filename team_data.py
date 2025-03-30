# European Soccer Teams by League
TEAMS = {
    "Premier League": [
        "Arsenal", "Aston Villa", "Brighton", "Burnley", "Chelsea", "Crystal Palace",
        "Everton", "Leeds", "Leicester", "Liverpool", "Manchester City", "Manchester United",
        "Newcastle", "Norwich", "Southampton", "Tottenham", "Watford", "West Ham", "Wolves"
    ],
    "La Liga": [
        "Athletic Bilbao", "Atletico Madrid", "Barcelona", "Cadiz", "Celta Vigo",
        "Elche", "Espanyol", "Getafe", "Granada", "Levante", "Mallorca", "Osasuna",
        "Real Betis", "Real Madrid", "Real Sociedad", "Sevilla", "Valencia", "Villarreal"
    ],
    "Bundesliga": [
        "Augsburg", "Bayer Leverkusen", "Bayern Munich", "Bochum", "Borussia Dortmund",
        "Borussia Monchengladbach", "Eintracht Frankfurt", "Freiburg", "Greuther Furth",
        "Hertha Berlin", "Hoffenheim", "Koln", "Mainz", "RB Leipzig", "Stuttgart",
        "Union Berlin", "Wolfsburg"
    ],
    "Serie A": [
        "Atalanta", "Bologna", "Cagliari", "Empoli", "Fiorentina", "Genoa",
        "Inter", "Juventus", "Lazio", "Milan", "Napoli", "Roma", "Salernitana",
        "Sampdoria", "Sassuolo", "Spezia", "Torino", "Udinese", "Venezia", "Verona"
    ],
    "Ligue 1": [
        "Angers", "Bordeaux", "Brest", "Clermont", "Lens", "Lille", "Lyon",
        "Marseille", "Metz", "Monaco", "Montpellier", "Nantes", "Nice", "Paris Saint-Germain",
        "Reims", "Rennes", "Saint-Etienne", "Strasbourg", "Troyes"
    ],
    "Eredivisie": [
        "Ajax", "AZ", "Cambuur", "Feyenoord", "Fortuna Sittard", "Go Ahead Eagles",
        "Groningen", "Heerenveen", "Heracles", "NEC", "PSV", "RKC Waalwijk", "Sparta Rotterdam",
        "Twente", "Utrecht", "Vitesse", "Willem II"
    ],
    "Primeira Liga": [
        "Arouca", "Benfica", "Boavista", "Braga", "Estoril", "Famalicao",
        "Gil Vicente", "Maritimo", "Moreirense", "Pacos de Ferreira", "Porto",
        "Santa Clara", "Sporting", "Tondela", "Vitoria Guimaraes", "Vizela"
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
        "Hammarby", "IFK Norrkoping", "Kalmar", "Malmo", "Mjallby", "Sirius",
        "Varberg", "Varnamo"
    ],
    "Greek Super League": [
        "AEK Athens", "Aris", "Asteras Tripolis", "Atromitos", "Ionikos", "Lamia",
        "Olympiacos", "Panathinaikos", "PAOK", "PAS Giannina", "Volos"
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
    ]
}

def get_team_league(team_name):
    """Find which league a team belongs to."""
    for league, teams in TEAMS.items():
        if team_name in teams:
            return league
    return "Other"

def get_all_teams():
    """Get a flat list of all teams."""
    return [team for teams in TEAMS.values() for team in teams]

def get_league_teams(league):
    """Get all teams in a specific league."""
    return TEAMS.get(league, []) 