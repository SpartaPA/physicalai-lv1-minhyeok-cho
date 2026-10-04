#ifndef MOTOR_HPP
#define MOTOR_HPP

class Motor
{
public:
    explicit Motor(int id);

    void set_speed(double rpm);
    double get_speed() const;
    void print_status() const;

private:
    int id_;
    double speed_rpm_;
};

#endif