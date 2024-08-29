import math
from sieve import gpu_acceleration as gpu


class Sieve:
    def __init__(self) -> None:
        pass

    def nth_prime(self, n: int, gpu_acceleration: bool = False) -> int:

        if n < 0:
            raise ValueError("n must be a positive integer.")

        if n == 0:
            return 2

        # https://en.wikipedia.org/wiki/Prime_number_theorem
        likely_upper_bound = int(n * math.log(n))

        # likely_upper_bound should include the primes but not guaranteed.
        # Let's add a small buffer to ensure we have enough range.
        boundary_fudge = int(n * math.log(math.log(n)))

        upper_bound = likely_upper_bound + boundary_fudge

        # Create a list, each element initialized to True (possible prime)
        # Since we are only marking odd numbers, we can use half the size
        sieve_size = (upper_bound // 2) + 1
        sieve = [True] * sieve_size
        count = 1

        if gpu_acceleration:
            sieve = gpu.initialize(sieve)

        for idx in range(1, sieve_size):
            # If we know it's not a prime, skip
            if not sieve[idx]:
                continue

            # The real number is twice the index + 1 (only using odd numbers)
            prime = 2 * idx + 1

            # If we found the nth prime, return it
            if count == n:
                return prime

            count += 1

            # Mark all multiples of the prime as False, using GPU if requested
            if gpu_acceleration:
                gpu.launch_kernel(sieve, prime, upper_bound)
            else:
                for multiple in range(prime * prime, upper_bound, prime * 2):
                    sieve[multiple // 2] = False

        # If no prime is found at this point, something went very wrong
        return None
