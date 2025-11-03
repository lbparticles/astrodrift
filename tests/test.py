from drift import integrate, potential, ic


def main():
    res = integrate(potential(), ic())
    res.run()


if __name__ == "__main__":
    main()
