#include "autodiff_test_2.h"

CostFunction::CostFunction(OBJ_FUNC_PTR running, OBJ_FUNC_PTR terminal)
{
    _running_cost = running;
    _terminal_cost = terminal;
}


void CostFunction::update_state(autodiff::VectorXdual& x, autodiff::VectorXdual& u)
{
    _u = u;
    _x = x;
}


VectorXd CostFunction::L_x()
{
    dual running_cost;
    return gradient(_running_cost, wrt(_x), at(_x, _u), running_cost);
}


VectorXd CostFunction::L_u()
{
    dual running_cost;
    return gradient(_running_cost, wrt(_u), at(_x, _u), running_cost);
}


VectorXd CostFunction::L_xx()
{
    dual running_cost;
    return gradient(_running_cost, wrt<2>(_x), at(_x, _u));
}


VectorXd CostFunction::L_ux()
{
    dual running_cost;
    return gradient(_running_cost, wrt(_u, _x), at(_x, _u), running_cost);
}


dual CostFunction::Lf()
{
    return _terminal_cost(_x, _u);
}


VectorXd CostFunction::Lf_x()
{
    dual running_cost;
    return gradient(_terminal_cost, wrt(_x), at(_x, _u), running_cost);
}


VectorXd CostFunction::Lf_xx()
{
    dual running_cost;
    return gradient(_terminal_cost, wrt<2>(_x), at(_x, _u), running_cost);
}

