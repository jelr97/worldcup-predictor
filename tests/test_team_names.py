from data.team_names import normalize, same_team


def test_aliases():
    assert same_team("USA", "United States")
    assert same_team("Korea Republic", "South Korea")
    assert same_team("Türkiye", "Turkey")
    assert same_team("Côte d'Ivoire", "Ivory Coast")
    assert same_team("Cabo Verde", "Cape Verde")
    assert same_team("Congo DR", "DR Congo")


def test_accents_and_case():
    assert normalize("  MÉXICO ") == "mexico"


def test_different_teams():
    assert not same_team("Brazil", "Argentina")
