from agent import mock_run


def main():
    research = mock_run("What is RAG?")
    assert "SUPERVISOR" in research
    assert "RESEARCH SPECIALIST" in research
    assert "rag" in research.lower()

    calc = mock_run("Calculate 240 / 40")
    assert "CALCULATOR SPECIALIST" in calc
    assert "6" in calc

    print("PASS: supervisor routed to research specialist")
    print(research)
    print("\nPASS: supervisor routed to calculator specialist")
    print(calc)


if __name__ == "__main__":
    main()
