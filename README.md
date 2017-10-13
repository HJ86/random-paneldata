Create random panel datasets
============================

Create random panel datasets, represented as `pandas.DataFrame`s, which confirm to a structure specified by a panel data equation.

Panel datasets often have several indices, such as an individual (often *i*) and a time period (often *t*). Each of these indices has a particular length, often referred to as either *N* or *T*, depending on the index.
Here, the lengths of all indices are specified by a parameter *N* which either gives, for each index, either a desired length or a list of values to use for that index.

## The equation
The equation is specified in a similar (but not identical) manner to that used in R.

- The left-hand side (LHS) and right-hand side (RHS) are seperated by a `~`
- Indices on variables are specified with an underscore, e.g `y_it` or `x_j`
- Operators `lag` and `diff` work on variables as follows: `operator[params](variable)`
- Parameter ranges are specified with a `:`, and are *inclusive*. Thus `0:2` means `0, 1 and 2`
example_eqn = 'y_ijt ~ lag[1:2](y_ijt) + lag[0:1](x_ijt) + z_i + alpha_j'
example_N = dict(i=4, j=4, t=list(range(2000, 2010)))
example_beta = [0.6, 0.2, 1, 1, 0.1, 0.5]

## The length(s) of the indices
The length of each index is specified by a single parameter `N`. This is a `dict` whose keys are the names of the indices (often `i`, `j`, `t` etc.). The values can be one of two things:

- An `int`, which specifies the desired length of the index. In this case, random string values of length 6 will be generated for each index level.
- A `list`, which become the levels of the index.

## The multiplying factors (betas)
Most panel data model specifications are additive with multiplying factors. These multiplying factors are specified using the `beta` parameter, in the order they appear in the equation, left to right.

A typical example might look like this:
`y_it = alpha_i + 0.5 * x1_it + 0.2 * x2_it + 0.8 * epsilon_it`

The multiplying factors (often known as "beta") implicitly include a 1 to multiply `alpha_i`. This must be specified *explicitly* here.

We can thus specify the above equation using:

```
>>>equation = 'y_it ~ alpha_i + x1_it + x2_it + epsilon_it'`
>>>beta = [1, 0.5, 0.2, 0.8]`
```

## Example

```
>>> import panel_data
>>> eqn = 'y_it ~ alpha + lag(x_it) + epsilon_i'
>>> N = dict(i=2, t=range(2010, 2013))
>>> beta = [1, 0.8, 0.2]
>>> df = panel_data.random_data_from_equation(eqn, N=N, beta=beta)
>>> df
                 y_it    alpha  lag(x_it)  epsilon_i
i      t
mnisem 2010 -1.048965 -0.97318        NaN  -0.378927
       2011 -0.601854 -0.97318   0.558889  -0.378927
       2012 -0.401793 -0.97318   0.808965  -0.378927
wqtkkn 2010 -1.370461 -0.97318        NaN  -1.986406
       2011 -1.522827 -0.97318  -0.190458  -1.986406
       2012 -1.671796 -0.97318  -0.376669  -1.986406
```
