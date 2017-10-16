from panel_data import *

var_x = Variable('x', 'it')
op_diff_x = Operator('diff', var_x)
op_lag2_diff_x = Operator('lag', op_diff_x, 2)
eq_element_op_lag2_diff_x = EquationElement(var_x, [op_lag2_diff_x])


