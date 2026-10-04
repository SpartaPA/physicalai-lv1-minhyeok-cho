#include "motor.hpp"

#include <iostream>

int main()
{
    Motor motor(42);

    motor.print_status();

    motor.set_speed(120.0);
    motor.print_status();

    std::cout << "Readback: "
              << motor.get_speed() << " RPM\n";

    return 0;
}