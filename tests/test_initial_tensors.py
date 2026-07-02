"""
Unit tests for initial_tensors module.
"""

import numpy as np
import pytest
import cytnx
from tn_hotrg.initial_tensors import ising_tensor, ising_tensor_directional, ising_tensor_z2


class TestIsingTensor:
    """Tests for ising_tensor function."""

    def test_tensor_shape(self):
        """Test that tensor has correct shape (2,2,2,2)."""
        T = ising_tensor(beta=1.0)
        assert T.shape() == [2, 2, 2, 2]

    def test_tensor_labels(self):
        """Test that tensor has correct leg labels."""
        T = ising_tensor(beta=1.0)
        labels = T.labels()
        assert labels == ["xi", "xo", "yi", "yo"]

    def test_tensor_name(self):
        """Test that tensor has correct name."""
        T = ising_tensor(beta=1.0)
        assert T.name() == "T"

    def test_tensor_values_beta_1(self):
        """Test tensor values at beta=1."""
        T = ising_tensor(beta=1.0)

        # Expected values computed analytically:
        # M = [[sqrt(cosh(1)), sqrt(sinh(1))],
        #      [sqrt(cosh(1)), -sqrt(sinh(1))]]
        # T[xi,xo,yi,yo] = M[0,xi]*M[0,xo]*M[0,yi]*M[0,yo]
        #                + M[1,xi]*M[1,xo]*M[1,yi]*M[1,yo]

        cosh_1 = np.cosh(1.0)
        sinh_1 = np.sinh(1.0)

        # T[0,0,0,0] = 2 * cosh(1)^2
        expected_0000 = 2.0 * cosh_1 ** 2
        actual_0000 = T.at(["xi", "xo", "yi", "yo"], [0, 0, 0, 0]).value
        assert np.isclose(actual_0000, expected_0000, rtol=1e-10)

        # T[1,1,1,1] = 2 * sinh(1)^2
        expected_1111 = 2.0 * sinh_1 ** 2
        actual_1111 = T.at(["xi", "xo", "yi", "yo"], [1, 1, 1, 1]).value
        assert np.isclose(actual_1111, expected_1111, rtol=1e-10)

        # T[0,0,1,1] = 2 * cosh(1) * sinh(1)
        expected_0011 = 2.0 * cosh_1 * sinh_1
        actual_0011 = T.at(["xi", "xo", "yi", "yo"], [0, 0, 1, 1]).value
        assert np.isclose(actual_0011, expected_0011, rtol=1e-10)

        # T[0,1,0,1] = 2 * cosh(1) * sinh(1)
        expected_0101 = 2.0 * cosh_1 * sinh_1
        actual_0101 = T.at(["xi", "xo", "yi", "yo"], [0, 1, 0, 1]).value
        assert np.isclose(actual_0101, expected_0101, rtol=1e-10)

        # T[0,0,0,1] = 0 (odd number of 1s, Z2 symmetry)
        actual_0001 = T.at(["xi", "xo", "yi", "yo"], [0, 0, 0, 1]).value
        assert np.isclose(actual_0001, 0.0, atol=1e-10)

        # T[1,0,0,0] = 0 (odd number of 1s, Z2 symmetry)
        actual_1000 = T.at(["xi", "xo", "yi", "yo"], [1, 0, 0, 0]).value
        assert np.isclose(actual_1000, 0.0, atol=1e-10)

    def test_z2_symmetry(self):
        """Test that tensor respects Z2 symmetry (odd parity elements are zero)."""
        T = ising_tensor(beta=1.0)

        # All elements with odd number of 1-indices should be zero
        for xi in [0, 1]:
            for xo in [0, 1]:
                for yi in [0, 1]:
                    for yo in [0, 1]:
                        parity = xi + xo + yi + yo
                        val = T.at(["xi", "xo", "yi", "yo"], [xi, xo, yi, yo]).value
                        if parity % 2 == 1:
                            assert np.isclose(val, 0.0, atol=1e-10), \
                                f"T[{xi},{xo},{yi},{yo}] should be 0, got {val}"

    def test_tensor_positivity(self):
        """Test that even parity elements are positive."""
        T = ising_tensor(beta=1.0)

        for xi in [0, 1]:
            for xo in [0, 1]:
                for yi in [0, 1]:
                    for yo in [0, 1]:
                        parity = xi + xo + yi + yo
                        val = T.at(["xi", "xo", "yi", "yo"], [xi, xo, yi, yo]).value
                        if parity % 2 == 0:
                            assert val > 0, \
                                f"T[{xi},{xo},{yi},{yo}] should be positive, got {val}"


class TestIsingTensorDirectional:
    """Tests for ising_tensor_directional function."""

    def test_tensor_shape(self):
        """Test that tensor has correct shape (2,2,2,2)."""
        T = ising_tensor_directional(beta=1.0)
        assert T.shape() == [2, 2, 2, 2]

    def test_tensor_labels(self):
        """Test that tensor has correct leg labels."""
        T = ising_tensor_directional(beta=1.0)
        labels = T.labels()
        assert labels == ["xi", "xo", "yi", "yo"]

    def test_tensor_name(self):
        """Test that tensor has correct name."""
        T = ising_tensor_directional(beta=1.0)
        assert T.name() == "T"

    def test_bond_directions(self):
        """Test that bonds have correct directions."""
        T = ising_tensor_directional(beta=1.0)
        bonds = T.bonds()

        # xi (index 0): BD_IN
        assert bonds[0].type() == cytnx.BD_IN, "xi should be BD_IN"
        # xo (index 1): BD_OUT
        assert bonds[1].type() == cytnx.BD_OUT, "xo should be BD_OUT"
        # yi (index 2): BD_IN
        assert bonds[2].type() == cytnx.BD_IN, "yi should be BD_IN"
        # yo (index 3): BD_OUT
        assert bonds[3].type() == cytnx.BD_OUT, "yo should be BD_OUT"

    def test_values_match_nondirectional(self):
        """Test that values match the non-directional version."""
        T_dir = ising_tensor_directional(beta=1.0)
        T_nondir = ising_tensor(beta=1.0)

        for xi in [0, 1]:
            for xo in [0, 1]:
                for yi in [0, 1]:
                    for yo in [0, 1]:
                        val_dir = T_dir.at(["xi", "xo", "yi", "yo"], [xi, xo, yi, yo]).value
                        val_nondir = T_nondir.at(["xi", "xo", "yi", "yo"], [xi, xo, yi, yo]).value
                        assert np.isclose(val_dir, val_nondir, rtol=1e-10), \
                            f"T[{xi},{xo},{yi},{yo}] mismatch: {val_dir} vs {val_nondir}"

    def test_tensor_values_beta_1(self):
        """Test tensor values at beta=1."""
        T = ising_tensor_directional(beta=1.0)

        cosh_1 = np.cosh(1.0)
        sinh_1 = np.sinh(1.0)

        # T[0,0,0,0] = 2 * cosh(1)^2
        expected_0000 = 2.0 * cosh_1 ** 2
        actual_0000 = T.at(["xi", "xo", "yi", "yo"], [0, 0, 0, 0]).value
        assert np.isclose(actual_0000, expected_0000, rtol=1e-10)

        # T[1,1,1,1] = 2 * sinh(1)^2
        expected_1111 = 2.0 * sinh_1 ** 2
        actual_1111 = T.at(["xi", "xo", "yi", "yo"], [1, 1, 1, 1]).value
        assert np.isclose(actual_1111, expected_1111, rtol=1e-10)

    def test_z2_symmetry(self):
        """Test that tensor respects Z2 symmetry."""
        T = ising_tensor_directional(beta=1.0)

        for xi in [0, 1]:
            for xo in [0, 1]:
                for yi in [0, 1]:
                    for yo in [0, 1]:
                        parity = xi + xo + yi + yo
                        val = T.at(["xi", "xo", "yi", "yo"], [xi, xo, yi, yo]).value
                        if parity % 2 == 1:
                            assert np.isclose(val, 0.0, atol=1e-10), \
                                f"T[{xi},{xo},{yi},{yo}] should be 0, got {val}"


@pytest.mark.xfail(
    reason="cytnx 1.1.0 get_qindices() returns ints; ising_tensor_z2 needs update",
    strict=False,
)
class TestIsingTensorZ2:
    """Tests for ising_tensor_z2 function."""

    def test_tensor_shape(self):
        """Test that tensor has correct shape (2,2,2,2)."""
        T = ising_tensor_z2(beta=1.0)
        assert T.shape() == [2, 2, 2, 2]

    def test_tensor_labels(self):
        """Test that tensor has correct leg labels."""
        T = ising_tensor_z2(beta=1.0)
        labels = T.labels()
        assert labels == ["xi", "xo", "yi", "yo"]

    def test_tensor_name(self):
        """Test that tensor has correct name."""
        T = ising_tensor_z2(beta=1.0)
        assert T.name() == "T"

    def test_bond_directions(self):
        """Test that bonds have correct directions."""
        T = ising_tensor_z2(beta=1.0)
        bonds = T.bonds()

        # xi (index 0): BD_IN
        assert bonds[0].type() == cytnx.BD_IN, "xi should be BD_IN"
        # xo (index 1): BD_OUT
        assert bonds[1].type() == cytnx.BD_OUT, "xo should be BD_OUT"
        # yi (index 2): BD_IN
        assert bonds[2].type() == cytnx.BD_IN, "yi should be BD_IN"
        # yo (index 3): BD_OUT
        assert bonds[3].type() == cytnx.BD_OUT, "yo should be BD_OUT"

    def test_bonds_have_z2_symmetry(self):
        """Test that bonds have Z2 symmetry."""
        T = ising_tensor_z2(beta=1.0)
        bonds = T.bonds()

        for i, bond in enumerate(bonds):
            syms = bond.syms()
            assert len(syms) == 1, f"Bond {i} should have exactly 1 symmetry"
            # Check it's Zn(2) symmetry
            assert syms[0].stype() == cytnx.Symmetry.Zn(2).stype(), \
                f"Bond {i} should have Z2 symmetry"

    def test_number_of_blocks(self):
        """Test that tensor has correct number of non-zero blocks."""
        T = ising_tensor_z2(beta=1.0)
        # Z2 symmetry: 8 valid blocks out of 16
        # (0,0,0,0), (0,0,1,1), (0,1,0,1), (0,1,1,0),
        # (1,0,0,1), (1,0,1,0), (1,1,0,0), (1,1,1,1)
        assert T.Nblocks() == 8

    def test_tensor_values_beta_1(self):
        """Test tensor values at beta=1."""
        T = ising_tensor_z2(beta=1.0)

        cosh_1 = np.cosh(1.0)
        sinh_1 = np.sinh(1.0)

        # Check block values by iterating over blocks
        for blk_idx in range(T.Nblocks()):
            qnums = T.get_qindices(blk_idx)
            q_xi, q_xo, q_yi, q_yo = [q[0] for q in qnums]
            parity = q_xi + q_xo + q_yi + q_yo

            val = T.get_block_(blk_idx)[0]

            if parity == 0:
                # (0,0,0,0): cosh(beta)
                assert np.isclose(val, cosh_1, rtol=1e-10), \
                    f"Block ({q_xi},{q_xo},{q_yi},{q_yo}) should be cosh(1)"
            elif parity == 4:
                # (1,1,1,1): sinh(beta)
                assert np.isclose(val, sinh_1, rtol=1e-10), \
                    f"Block ({q_xi},{q_xo},{q_yi},{q_yo}) should be sinh(1)"
            else:
                # Mixed parity: sqrt(cosh * sinh)
                expected = np.sqrt(cosh_1 * sinh_1)
                assert np.isclose(val, expected, rtol=1e-10), \
                    f"Block ({q_xi},{q_xo},{q_yi},{q_yo}) should be sqrt(cosh*sinh)"

    def test_all_block_values_positive(self):
        """Test that all block values are positive."""
        T = ising_tensor_z2(beta=1.0)

        for blk_idx in range(T.Nblocks()):
            val = T.get_block_(blk_idx)[0]
            assert val > 0, f"Block {blk_idx} should be positive, got {val}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
