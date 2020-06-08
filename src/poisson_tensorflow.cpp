#include "rng.hpp"

template <class T = int>
T RNG::rpois(const size_t thread_idx, double lambda) { 
    T x = 0;
    if (lambda < 10) {
        // Knuth's algorithm for generating Poisson random variates.
        // Given a Poisson process, the time between events is exponentially
        // distributed. If we have a Poisson process with rate lambda, then,
        // the time between events is distributed Exp(lambda). If X ~
        // Uniform(0, 1), then Y ~ Exp(lambda), where Y = -log(X) / lambda.
        // Thus to simulate a Poisson draw, we can draw X_i ~ Exp(lambda),
        // and N ~ Poisson(lambda), where N is the least number such that
        // \sum_i^N X_i > 1.
        const double exp_neg_rate = exp(-lambda);

        double prod = 1;

        // Keep trying until we surpass e^(-rate). This will take
        // expected time proportional to rate.
        while (true) {
            double u = this->runif(thread_idx);
            prod = prod * u;
            if (prod <= exp_neg_rate &&
                x <= std::numeric_limits<T>::max()) {
                break;
            }
            x += 1;
        }
    } else {
        // Transformed rejection due to Hormann.
        //
        // Given a CDF F(x), and G(x), a dominating distribution chosen such
        // that it is close to the inverse CDF F^-1(x), compute the following
        // steps:
        //
        // 1) Generate U and V, two independent random variates. Set U = U - 0.5
        // (this step isn't strictly necessary, but is done to make some
        // calculations symmetric and convenient. Henceforth, G is defined on
        // [-0.5, 0.5]).
        //
        // 2) If V <= alpha * F'(G(U)) * G'(U), return floor(G(U)), else return
        // to step 1. alpha is the acceptance probability of the rejection
        // algorithm.
        //
        // For more details on transformed rejection, see:
        // http://citeseer.ist.psu.edu/viewdoc/citations;jsessionid=1BEB35946CC807879F55D42512E5490C?doi=10.1.1.48.3054.
        //
        // The dominating distribution in this case:
        //
        // G(u) = (2 * a / (2 - |u|) + b) * u + c

        const double log_rate = log(lambda);

        // Constants used to define the dominating distribution. Names taken
        // from Hormann's paper. Constants were chosen to define the tightest
        // G(u) for the inverse Poisson CDF.
        const double b = 0.931 + 2.53 * sqrt(lambda);
        const double a = -0.059 + 0.02483 * b;

        // This is the inverse acceptance rate. At a minimum (when rate = 10),
        // this corresponds to ~75% acceptance. As the rate becomes larger, this
        // approaches ~89%.
        const double inv_alpha = 1.1239 + 1.1328 / (b - 3.4);

        while (true) {
            double u = this->runif(thread_idx);
            u -= 0.5;
            double v = this->runif(thread_idx);

            double u_shifted = 0.5 - abs(u);
            T k = floor((2 * a / u_shifted + b) * u + lambda +
                                        0.43);

            if (k > std::numeric_limits<T>::max()) {
                // retry in case of overflow.
                continue;
            }

            // When alpha * f(G(U)) * G'(U) is close to 1, it is possible to
            // find a rectangle (-u_r, u_r) x (0, v_r) under the curve, such
            // that if v <= v_r and |u| <= u_r, then we can accept.
            // Here v_r = 0.9227 - 3.6224 / (b - 2) and u_r = 0.43.
            if (u_shifted >= 0.07 &&
                v <= 0.9277 - 3.6224 / (b - 2)) {
                x = k;
                break;
            }

            if (k < 0 || (u_shifted < 0.013 && v > u_shifted)) {
                continue;
            }

            // The expression below is equivalent to the computation of step 2)
            // in transformed rejection (v <= alpha * F'(G(u)) * G'(u)).
            double s = log(v * inv_alpha / (a / (u_shifted * u_shifted) + b));
            double t = -lambda + k * log_rate - lgamma(k + 1);
            if (s <= t) {
                x = k;
                break;
            }
        }
    }
    return(x);
}

template int RNG::rpois<int>(const size_t thread_idx, double lambda); 
