import unittest
from sieve import Sieve


class SieveTest(unittest.TestCase):

    def test_sieve_nth_prime_with_gpu(self) -> None:
        sieve = Sieve()
        self.assertEqual(2, sieve.nth_prime(0, True))
        self.assertEqual(71, sieve.nth_prime(19, True))
        self.assertEqual(541, sieve.nth_prime(99, True))
        self.assertEqual(3581, sieve.nth_prime(500, True))
        self.assertEqual(7793, sieve.nth_prime(986, True))
        # self.assertEqual(15485867, sieve.nth_prime(1000000, True))
        # self.assertEqual(179424691, sieve.nth_prime(10000000, True))
        # self.assertEqual(2038074751, sieve.nth_prime(100000000, True))
