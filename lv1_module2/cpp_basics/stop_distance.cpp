#include <cmath>
#include <iomanip>
#include <iostream>

int main()
{
    constexpr double g = 9.81;

    double velocity = 0.0;
    double friction = 0.0;

    std::cout << "Enter speed (m/s) and friction coefficient: ";

    if (!(std::cin >> velocity >> friction)) {
        std::cerr << "Error: enter two numbers.\n";
        return 1;
    }

    if (!std::isfinite(velocity) || !std::isfinite(friction)
        || velocity < 0.0 || friction <= 0.0) {
        std::cerr << "Error: speed must be nonnegative "
                     "and friction must be positive and finite.\n";
        return 1;
    }

    const double distance =
        velocity * velocity / (2.0 * friction * g);

    std::cout << std::fixed << std::setprecision(3)
              << "Stopping distance: " << distance << " m\n";

    return 0;
}