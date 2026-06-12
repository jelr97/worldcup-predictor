from data.team_names import normalize, same_team


def test_aliases():
    assert same_team("USA", "United States")
    assert same_team("Korea Republic", "South Korea")
    assert same_team("Türkiye", "Turkey")
    assert same_team("Côte d'Ivoire", "Ivory Coast")
    assert same_team("Cabo Verde", "Cape Verde")
    assert same_team("Congo DR", "DR Congo")
    assert same_team("Bosnia & Herzegovina", "Bosnia and Herzegovina")


def test_accents_and_case():
    assert normalize("  MÉXICO ") == "mexico"


def test_different_teams():
    assert not same_team("Brazil", "Argentina")


# ── Spanish aliases added for experts ensemble ────────────────────────────────

def test_spanish_aliases_resolve():
    """All Spanish names from Poia.xlsx must match their canonical fixture name."""
    # Direct Spanish -> canonical checks
    assert same_team("Alemania", "Germany")
    assert same_team("Arabia Saudita", "Saudi Arabia")
    assert same_team("Argelia", "Algeria")
    assert same_team("Bélgica", "Belgium")
    assert same_team("Bosnia y Herzegovina", "Bosnia and Herzegovina")
    assert same_team("Brasil", "Brazil")
    assert same_team("Chequia", "Czechia")
    assert same_team("Corea del Sur", "Korea Republic")
    assert same_team("Costa de Marfil", "Côte d'Ivoire")
    assert same_team("Croacia", "Croatia")
    assert same_team("Curazao", "Curaçao")
    assert same_team("Egipto", "Egypt")
    assert same_team("Escocia", "Scotland")
    assert same_team("España", "Spain")
    assert same_team("Estados Unidos", "USA")
    assert same_team("Francia", "France")
    assert same_team("Inglaterra", "England")
    assert same_team("Irak", "Iraq")
    assert same_team("Irán", "IR Iran")
    assert same_team("Japón", "Japan")
    assert same_team("Jordania", "Jordan")
    assert same_team("Marruecos", "Morocco")
    assert same_team("Noruega", "Norway")
    assert same_team("Nueva Zelanda", "New Zealand")
    assert same_team("Países Bajos", "Netherlands")
    assert same_team("RD Congo", "Congo DR")
    assert same_team("Sudáfrica", "South Africa")
    assert same_team("Suecia", "Sweden")
    assert same_team("Suiza", "Switzerland")
    assert same_team("Túnez", "Tunisia")
    assert same_team("Turquía", "Türkiye")
