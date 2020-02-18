#include <iostream>
#include "autodiff/forward.hpp"
#include "autodiff/reverse.hpp"
#include "../utils/time_bench.h"

using namespace autodiff;

template <typename T>
inline static T sinus(T& x)
{
    return sin(x);
}

static inline var sinus(var& x)
{
    return sin(x);
}

static dual sinus(dual& x)
{
    return sin(x);
}

template <typename T>
inline static T first_order_finite_diff(T& around, T& step)
{
    T perturbation_p = around + step; T perturbation_n = around - step;
    return (sinus(perturbation_p) - sinus(perturbation_n))/(2 * step);
}

template dual sinus<dual>(dual& x);
template var sinus<var>(var& x);

template dual first_order_finite_diff<dual>(dual& around, dual& step);
template var first_order_finite_diff<var>(var& around, var& step);

int main()
{
    TimeBench timer;
    dual input_d = 0.12; var input_v = 0.12;
    dual perturb_d = 0.01; var perturb_v = 0.01;
    dual s_d  = sinus(input_d);
    var s_v   = sinus(input_v);
    dual dsdx_finite = first_order_finite_diff(input_d, perturb_d);
    dual dsdx_autodiff_for = derivative(sinus<dual>, wrt(input_d), at(input_d));
    Derivatives dud = derivatives(s_v);
    var dsdx_autodiff_rev = dud(input_v);

    using namespace std;
    cout << "ds/dx real     = " << cos(0.12) << "\n";         // print the evaluated output u
    cout << "ds/dx finite   = " <<  dsdx_finite << "\n";  // print the evaluated derivative du/dx
    cout << "ds/dx autodiff forward  =  " << dsdx_autodiff_for << "\n";  // print the evaluated derivative du/dy
    cout << "ds/dx autodiff reverse  =  " << dsdx_autodiff_rev << "\n";  // print the evaluated derivative du/dy
}


