#include "motor.hpp"

#include <iostream>

Motor::Motor(int id)
    : id_(id), speed_rpm_(0.0)
{
}

void Motor::set_speed(double rpm)
{
    speed_rpm_ = rpm;
}

double Motor::get_speed() const
{
    return speed_rpm_;
}

void Motor::print_status() const
{
    std::cout << "Motor ID" << id_
              << ": " << speed_rpm_ << " RPM\n";
}