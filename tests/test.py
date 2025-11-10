import drift as dft


def main():
    gal = dft.bg_feature()
    gmc = dft.part_group([])
    iso = dft.test_group([])
    sim = dft.simulation([gal, gmc, iso])
    _dfs = sim.run()


if __name__ == "__main__":
    main()
