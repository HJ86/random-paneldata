"""
panel_data.py

Create random panel data sets according to a data generating process specified in equation form, with dimensions specified either by length, in which case the dimension levels are randomly generated, or with specific levels.

>>> import numpy as np; np.random.seed(0)
>>> equation = "y_it ~ lag(y_it) + lag[0:2](x_it) + alpha_i"
>>> N = dict(i=5, t=range(2000, 2010))
>>> beta = [0.8, 0.6, 0.4, 0.2, 0.1]
>>> panel_data.random_data_from_equation(equation, N, beta).head(15)
                 y_it   alpha_i  lag(x_it)  lag(x_it)2  lag(y_it)      x_it
i      t
mpvadd 2000  0.284061  0.177426        NaN         NaN        NaN  0.443863
       2001  0.622741  0.177426   0.443863         NaN   0.284061  0.333674
       2002  1.634625  0.177426   0.333674    0.443863   0.622741  1.494079
       2003  1.866714  0.177426   1.494079    0.333674   1.634625 -0.205158
       2004  1.915707  0.177426  -0.205158    1.494079   1.866714  0.313068
       2005  1.122046  0.177426   0.313068   -0.205158   1.915707 -0.854096
       2006 -0.895439  0.177426  -0.854096    0.313068   1.122046 -2.552990
       2007 -1.498452  0.177426  -2.552990   -0.854096  -0.895439  0.653619
       2008 -0.911508  0.177426   0.653619   -2.552990  -1.498452  0.864436
       2009 -0.680265  0.177426   0.864436    0.653619  -0.911508 -0.742165
hjtvse 2000  1.321675 -0.401781        NaN         NaN        NaN  2.269755
       2001  1.052444 -0.401781   2.269755         NaN   1.321675 -1.454366
       2002  0.701437 -0.401781  -1.454366    2.269755   1.052444  0.045759
       2003  0.136091 -0.401781   0.045759   -1.454366   0.701437 -0.187184
       2004  0.922641 -0.401781  -0.187184    0.045759   0.136091  1.532779
"""

import numpy as np
import pandas as pd
from collections import namedtuple
import string
import itertools

class Variable:
    """
    A Variable, like x_ij, with a name and an optional list of
    indices
    >>> x = Variable('x', 'ij')
    >>> print(x)
    x_ij
    >>> y = Variable('y')
    >>> y
    y
    """
    def __init__(self, name, indices=None):
        self.name = name
        self.indices = indices
        
    def __repr__(self):
        if not self.indices:
            return self.name
        return "_".join((self.name, "".join(self.indices)))

class Operator:
    """
    An Operator, like lag(.) or diff[2](.) or lag[0:2](.)
    >>> print(Operator('lag', 'y'))
    lag(y)
    >>> print(Operator('diff', 'x', 2))
    diff[2](x)
    >>> print(Operator('lag', 'z', '0:2'))
    lag[0:2](z)
    """
    def __init__(self, name, argument, params=None):
        self.name = name
        self.argument = argument
        self.params = params
        
    def __repr__(self):
        if not self.params:
            return "{0}({1})".format(self.name, self.argument)
        else:
            return "{0}[{1}]({2})".format(self.name,
                                         self.params, self.argument)

class EquationElement:
    """
    An element of an additive equation, such as lag[2](y_ij). It consists
    of a variable and an optioanl list of associated operators.

    >>> var = Variable('y', 'ij')
    >>> op = Operator('lag', var, 2)
    >>> print(EquationElement(var, [op]))
    lag[2](y_ij)
    """
    def __init__(self, variable, operators=None):
        self.variable = variable
        self.operators = operators

    def __repr__(self):
        # TODO: This doesn't work for nested operators!
        if len(self.operators) == 0:
            return str(self.variable)
        return str(self.operators[0])
        
def _extract_element(op, names=[]):
    """
    Extract an EquationElement and its operators from within
    a possibly nested set of Operators   
    """
    # TODO: This seems a bit circular: we're creating EquationElements
    #       with variables inside, then extracting the variables out of them
    #       Is that really necessary?
    # TODO: What is `names`? What the hell is this function actually for?
    try:
        # Recurse if `op` looks like an Operator
        return _extract_element(op.argument, [op] + names)
    except AttributeError:
        # `op` doesn't look like an Operator. Create an EquationElement
        return EquationElement(variable=_variable(op), operators=names)

def _variable(el):
    """Extract name and indices from string `el`"""
    el = str(el).split('_')
    name = el[0]
    try:
        indices = el[1]
    except IndexError:
        # No indices
        indices = ""
    return Variable(name=name, indices=tuple(indices))

def _get_params(op):
    """
    Extract any parameters from an operator (expressed using
    square brackets after the parameter name and before the
    parentheses)
    """
    param_start = op.find('[')
    if param_start > 0:
        params = op[param_start + 1:op.rfind(']')]
        op = op[:param_start]
        return (op, params)
    else:
        return (op, "")
    
def _find_operators(el):
    """Recursively create Operator name/argument pairs from string `el`"""
    el = str(el)
    # First open parenthesis
    arg_start = el.find('(')
    if arg_start < 0:
        # No parentheses. Return original string
        return el
    # Last closed parenthesis
    arg_end = el.rfind((')'))
    operator = el[:arg_start]
    # Recurse until there's no more parens.
    argument = _find_operators(el[arg_start + 1:arg_end])
    operator, params = _get_params(operator)
    return Operator(name=operator, argument=argument, params=params)
    
def _parse_equation_element(el):
    """Create an EquationElement from el"""
    el = str(el)
    # Are there any operators like lag() log() etc.?
    operators = _find_operators(el)
    return _extract_element(operators)
        
def parse_equation(eq):
    """
    Create a (LHS, RHS) tuple where LHS and RHS are the two sides of
    an equation.
    
    Equations are split into LHS and RHS using '~' and the elements of
    the RHS are split using '+'.
    
    Each element of the equation can optionally have indices (like 'i' or 't')
    and operators (like lag() or diff() etc.).
    """
    e = str(eq).split('~')
    lhs = e[0].strip()
    lhs = _parse_equation_element(lhs)
    try:
        rhs = e[1].strip()
    except IndexError:
        raise TypeError('Equation must contain "~"')
    # Split the RHS on +
    rhs = [el.strip() for el in rhs.split('+')]
    # Parse each element as an EquationElement
    rhs = [_parse_equation_element(el) for el in rhs]
    return (lhs, rhs)


test_eq = "diff(y_it) ~ alpha + lag[1/2](diff(y_it)) + mu_i + x1_it + log(x1_it) + x2_t"

def _squeeze(tup):
    """Reduce a tuple to its value if it's a singleton"""
    if len(tup) == 1:
        return tup[0]
    return tup

def _product(iterables):
    """A list of `itertools._product` of `iterables`"""
    try:
        return [_squeeze(x) for x in itertools.product(*iterables)]
    except TypeError:
        return iterables

def _tuples_to_columns(tuples, names):
    """
    Zip `tuples` and assign `names` to each element.
    
    Examples
    ========
    >>> x = [('a', 'c'), ('b', 'd')]
    >>> _tuples_to_columns(x, ('A', 'B'))
    {'A': ('a', 'b'), 'B': ('c', 'd')}
    >>> x = ['a', 'b', 'c']
    >>> _tuples_to_columns(x, 'i')
    {'i': ('a', 'b', 'c')}
    """
    out = {}
    if len(names) == 1:
        out[names[0]] = tuples
        return out
    for i, x in enumerate(zip(*tuples)):
        out[names[i]] = x
    return out

def _assign_index_levels_to_data(levels, data):
    """
    Assign an index level to each element of data, according
    to the index (or indices) of the relevant Variable.
    
    `data` is a dict whose values are the data to be assigned
    and whose keys are the Variable that data represents.
    
    `levels` is a dict, keyed on the name of the index,
    of level values for that index.
    """
    out = {}
    for var, d in data.items():
        name = var.name
        indices = _squeeze(var.indices)
        idx_values = _product([levels[i] for i in indices])
        out[var] = {idx_values[i]: v for i, v in enumerate(d)}
    return out

def _data_columns(assigned_data, index_columns):
    """
    Create columns of data which align with `index_columns`
    
    Each element of `assigned_data` is a dict of index/value
    pairs, and `assigned_data` is itself a dict, keyed on
    Variable
    """
    df = {}
    for var, d in assigned_data.items():
        indices = _squeeze(var.indices)
        # What do we do if `indices` is empty??
        if len(indices) == 0:
            df[var.name] = d[()]
        else:
            idx = [_squeeze(x) for x in zip(*[index_columns[i] for i in indices])]
            df[var.name] = [d[i] for i in idx]
    return df

def levels_and_data_to_data_frame(levels, data):
    """
    Create a `pandas.DataFrame` with every combination
    of `levels` and data corresponding to the
    values of `data` according to which Variable the
    values are keyed on
    
    `levels` is a dict whose keys are the names
    of the levels
    """
    # Assign data values to index levels
    assigned_data = _assign_index_levels_to_data(levels, data)

    # Create the index with every possible combination
    # of levels
    index = _product(levels.values())

    # Creating a dict of columns of the index
    # with the level names as the keys
    level_names = list(levels)
    index_columns = _tuples_to_columns(index, names=level_names)

    columns = _data_columns(assigned_data, index_columns)
    if len(level_names) > 1:
        index = pd.MultiIndex.from_tuples(index, names=level_names)
    else:
        index = pd.Index(index, name=level_names[0])
    return pd.DataFrame(columns, index=index)     


def _multiply_list_elements(l, start=1):
    """
    Apply the * operator to all elements in l
    """
    result = start
    for x in l:
        result *= x
    return result

def _random_string(length):
    """
    A random string of lower-case letters of length `length`
    """
    letters = list(string.ascii_lowercase)
    random_letters = np.random.choice(letters, length)
    return "".join(random_letters)
    
def _unique_variables(els):
    """A unique set of `Variable`s from the `EquationElements` in `els`"""
    return set([el.variable for el in els])
    
def _list_length_or_number(x):
    """x is either a list, in which case get its length, or just a number"""
    try:
        # Assume x is a list
        return len(x)
    except TypeError:
        # If it's not, it's just a length
        return x
    
def _required_variable_length(variable, N):
    """
    Get the required length of the a variable with `indices` given `N`.
    
    This will depend on the indices (if any) that were passed, and the
    lengths of those indices (if specified) in `N`.
    
    `N` is either an int, a dict of ints or a dict of lists.
    """
    try:
        indices = variable.indices
    except AttributeError:
        # variable has no indices. This means it is a singleton
        return 1
    try:
#         length = 1
#         for n in [lengths[ix] for ix in indices]:
#             length *= _list_length_or_number(n)
        lengths = {k: _list_length_or_number(v) for k, v in N.items()}
        length = _multiply_list_elements([lengths[ix] for ix in indices])
        return length
    except TypeError:
        return int(lengths)
    except KeyError:
        raise KeyError("N does not contain the correct keys: {0}".format(indices))

def _required_data_length(N):
    """
    The length of the whole data set, given the lengths in N
    """
    lengths = [_list_length_or_number(n) for n in N.values()]
    return _multiply_list_elements(lengths)
    
def _get_indices(vars):
    """Extract all (unique) indices from all variables in `vars`"""
    indices = set()
    for ix in [set(var.indices) for var in vars]:
        indices = indices.union(set(ix))
    return list(indices)

def _create_levels(indices, N, length=6):
    """
    Create distinct levels for each of the set of `indices`
    according to `N`.
    
    If the relevant element of `N` is a list, these become
    the levels, otherwise the levels are a random string
    of length `length`.
    """
    level_values = {}
    for i in indices:
        n = N[i]
        try:
            lvl = [_random_string(length) for _ in range(n)]
        except TypeError:
            lvl = n
        level_values[i] = lvl
    return level_values
    
def independent_vars(lhs, rhs):
    """
    Extract a list of unique independent Variables from the
    EquationElements in `lhs` and `rhs`
    """
    return [v for v in _unique_variables(rhs) if v.name != lhs.variable.name]

def create_independent_data(variables, N):
    data = {}
    for variable in variables:
        length = _required_variable_length(variable, N)
        array = np.random.randn(length)
        data[variable] = array
    return data    

def random_data_from_equation(equation, N, beta, mu=None, sigma=None):
    """
    Create random data which abides by the structure given by `equation`
    with size(s) N.
    
    `equation` is a string which specifies the structure of the data
    along the following lines: y_it ~ lag(y_it) + x_it + mu_i + eps_it.
    
    Each equation element has a name (`x`, `y` etc.), optional indices
    (the `_it` part) and optional operators (`lag()`, `log()`, `diff()` etc.)
    
    If there are multiple dimensions (indices) specified in `equation`,
    `N` can be a dict specifying either the number of observations in 
    each index, or a list of values to use for that index. Otherwise, N
    is just a number, specifying the length of the data set.
    
    Where `N` is a number of observations, random values will be created for
    the index levels. Otherwise, the values given in `N` will be used for
    the index levels.
    
    `beta` is a list of multipliers to apply, in order, to the additive
    terms on the RHS of the equation.
    
    `mu` and `sigma` are currently not used
    """
    lhs, rhs = parse_equation(equation)
    rhs_variables = independent_vars(lhs, rhs)
    indep_vars = independent_vars(lhs, rhs)
    indep_elements = [el for el in rhs if el.variable in indep_vars]
    ## dep_elements = [el for el in rhs if el.variable.name == lhs.variable.name]
    # Find all the indices of the independent variables
    # e.g x_it + z_j -> (i, j, t)
    indices = _get_indices(rhs_variables)
    # Create the index levels, using N. These are either
    # given explicitly in N, or are created as a list of
    # random strings of the length specified by N.
    levels = _create_levels(indices, N)
    # Create a set of random values for each of the RHS variables. 
    # N is used to calculate the required length
    random_values = create_independent_data(rhs_variables, N)
    # Convert the resulting random values to a pandas.DataFrame
    # with index calculated according to `levels`.
    random_data = levels_and_data_to_data_frame(levels, random_values)
    # Create the independent variables DataFrame
    indep_data = populate_equation_elements(indep_elements, random_data)
    # Create the dependent variable(s)
    dep_vars, new_indep_data = create_dependent_variables(lhs, rhs, indep_data, beta)
    # Attach them to the other variables
    #from IPython.core.debugger import set_trace; set_trace()
    dep_vars = pd.DataFrame(dep_vars, index=indep_data.index, columns=[str(lhs)])
    return pd.concat([dep_vars, new_indep_data], axis=1)

def _dependendent_var_elements(lhs, rhs):
    """
    A list of EquationElements in `rhs` whose variables match that of `lhs`
    """
    return [el for el in rhs if el.variable == lhs.variable]

def _parameter_is_range(param):
    """Does the parameter represent a range of values?"""
    return param.contains(':')

def _range_from_parameter(param):
    """
    range(i, j) if `param` is "i:j", `param` if it's a number,
    or just 1 if `param` is empty.
    """
    param = param.split(':')
    if param[0] == '':
        return [1]
    if len(param) == 1:
        return param
    return range(int(param[0]), int(param[1]) + 1)

def _operator_name(operator, parameter):
    """Descriptive name of `operator`"""
    col_name = "{0}({1})".format(operator.name, operator.argument)
    if parameter == 0:
        col_name = operator.argument
    elif parameter > 1:
        col_name += '{}'.format(parameter)
    return col_name

def _apply_operator_with_param(s, operator, parameter):
    """
    Apply `operator` to pandas.Series or pandas.GroupBy `s`
    with a parameter
    """
    if operator.name == 'lag':
        # TODO: How do we know which index is time?
        s = s.shift(parameter)
    elif operator.name == 'diff':
        s = s.diff(parameter)
    return s

def _apply_operator(series, operator):
    """Apply `operator` to `series`"""
    result = []
    # TODO: How do we know which element is time? For now, guess `t`
    groupby = [c for c in series.index.names if c != 't']
    g = series.groupby(level=groupby)
    param_range = _range_from_parameter(operator.params)
    for param in param_range:
        s = _apply_operator_with_param(g, operator, param)
        col_name = _operator_name(operator, param)
        s.name = col_name
        result.append(s)
    return pd.concat(result, axis=1)

def _apply_all_operators(element, series):
    """Apply all operators (if any) to EquationElement `element`"""
    # TODO: Nested operators don't work yet
    if len(element.operators) == 0:
        # There are no operators. The result
        # is just the data itself
        series.name = str(element.variable)
        return [series]
    else:
        operated = []
        for op in element.operators:
            result = _apply_operator(series, op)
            operated.append(result)
        return operated

def populate_equation_elements(elements, data):
    """
    Create the data associated with the EquationElements in
    `elements`. This involves applying any `Operators` to
    the pandas.DataFrame in `data`
    """
    results = []
    for el in elements:
        series = data[el.variable.name]
        results.extend(_apply_all_operators(el, series))
    return pd.concat(results, axis=1)
    

class PositiveIndexList(list):
    """A list which does not accept a negative index"""
    def __getitem__(self, n):
        if n < 0:
            raise IndexError("This list does not accept a negative index")
        return list.__getitem__(self, n)

def _variables_are_equal(el1, el2):
    """Compare the variables in two equation elements"""
    return el1.variable.name == el2.variable.name

def create_dependent_variables(lhs_element, rhs_elements,
                               independent_data, beta):
    """
    Generate the data associated with each EquationElement in
    `dependent_elements`, using the data from the independent variables
    and the various multipliers in `beta`.

    TODO: This function is terrible. It could do with a complete re-write.
    """
    # I think this has to happen row-by-row because of the possibility
    # of a dependence of y_it on y_i(t-1)
    #dependent_data = PositiveIndexList()
    # Start by creating an empyt Series with the same index as
    # the independent data we've been passed.
    dependent_data = pd.Series(index=independent_data.index)
    rows = []
    for idx, row in independent_data.iterrows():
        # Calculate the value of each of the RHS elements
        element_values = {}
        for j, el in enumerate(rhs_elements):
            if _variables_are_equal(el, lhs_element):
                # We are dealing with a dependent variable
                if len(el.operators) == 0:
                    # No operators on dependent element on the RHS!
                    raise AttributeError('Dependent elements on the right-hand side must have an operator')
                for op in el.operators:
                    for param in _range_from_parameter(op.params):
                        operator_name = _operator_name(op, param)
                        if op.name == 'lag':
                            # TODO: this is a fairly dumb way of lagging. Can we improve?
                            lagged_index = []
                            for i, idx_name in enumerate(dependent_data.index.names):
                                idx_value = idx[i]
                                # TODO: How do we know that the time index is called `t`??
                                if idx_name == 't':
                                    lagged_index.append(idx_value - param)
                                else:
                                    lagged_index.append(idx_value)
                            try:
                                lagged_index = tuple(lagged_index) # Required for index slicing
                                res = dependent_data.loc[lagged_index]
                                #from IPython.core.debugger import set_trace; set_trace()
                                # Add the operated-upon dep. var to the independent data
                                row.loc[operator_name] = res
                            except KeyError:
                                # We're trying to access values of dependent_data which don't exist
                                res = 0
                            element_values[operator_name] = res
                        else:
                            raise NotImplementedError("Operators other than lag not implemented")
            else:
                # This element is an independent variable
                if len(el.operators) == 0:
                    var_name = str(el.variable)
                    res = row[var_name]
                    element_values[var_name] = res
                else:
                    for op in el.operators:
                        for param in _range_from_parameter(op.params):
                            operator_name = _operator_name(op, param)
                            res = row[operator_name]
                            element_values[operator_name] = res
        # Now process the independent variables' operators and sum
        x_beta = []
        for i, value in enumerate(element_values.values()):
            try:
                x_beta.append(value * beta[i])
            except:
                x_beta.append(0)
        #from IPython.core.debugger import set_trace; set_trace()
        dependent_data.loc[idx] = np.nansum(x_beta)
        rows.append(row)
    return dependent_data, pd.DataFrame(rows)

example_eqn = 'y_ijt ~ lag[1:2](y_ijt) + lag[0:1](x_ijt) + z_i + alpha_j'
example_N = dict(i=4, j=4, t=list(range(2000, 2010)))
example_beta = [0.6, 0.2, 1, 1, 0.1, 0.5]

if __name__ == '__main__':
    np.random.seed(0)
    df = random_data_from_equation(eqn, N=N, beta=beta)

def _a_test():
    levels = {}
    levels['i'] = ['a', 'b']
    levels['t'] = [1,2,3]

    variables = {}
    variables['x'] = Variable('x', 'i')
    variables['µ'] = Variable('µ', 't')
    variables['?'] = Variable('?', ('i', 't'))

    data = {}
    data[variables['x']] = ['y', 'z']
    data[variables['µ']] = ['m', 'n', 'o']
    data[variables['?']] = ['c', 'd', 'e', 'f', 'g', 'h']

    df = levels_and_data_to_data_frame(levels, data)
    df
