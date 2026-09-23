# Common methods for CompositePotential-type classes


class baseCompositePotential:
    def __len__(self):
        """Return the number of potentials in the composite."""
        return len(self._potlist)



    def __iter__(self):
        """Iterate over potentials in the composite."""
        return iter(self._potlist)






