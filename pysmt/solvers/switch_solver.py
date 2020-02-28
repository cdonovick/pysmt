import functools as ft
import itertools as it
import operator
from pysmt.exceptions import SolverAPINotFound

try:
    import smt_switch as ss
except ImportError:
    raise SolverAPINotFound

from pysmt.solvers.solver import (IncrementalTrackingSolver, UnsatCoreSolver,
                                  Converter, SolverOptions)
from pysmt.solvers.smtlib import SmtLibBasicSolver, SmtLibIgnoreMixin
from pysmt.solvers.eager import EagerModel
from pysmt.walkers import DagWalker
from pysmt.exceptions import (SolverReturnedUnknownResultError,
                              ConvertExpressionError, PysmtValueError)
from pysmt.decorators import clear_pending_pop, catch_conversion_error
from pysmt.logics import QF_BV, QF_UFBV, QF_ABV, QF_AUFBV, QF_AX
from pysmt.constants import to_python_integer


class SwitchOptions(SolverOptions):
    def __call__(self, solver):
        if self.generate_models:
            solver.solver.set_opt('produce-models', 'true')
        if self.incremental:
            solver.solver.set_opt('incremental', 'true')

class SwitchSolver(IncrementalTrackingSolver,
                   SmtLibBasicSolver,
                   SmtLibIgnoreMixin):
    LOGICS = [QF_BV, QF_UFBV,]# QF_ABV, QF_AUFBV, QF_AX]

    OptionsClass = SwitchOptions

    def __init__(self, environment, logic, **options):
        IncrementalTrackingSolver.__init__(self,
                        environment=environment,
                        logic=logic,
                        **options)

        self.solver = ss.create_msat_solver()
        self.options(self)
        self.converter = SwitchConverter(environment,  self.solver)
        self.mgr = environment.formula_manager


    def get_model(self):
        assignment = {}
        for s in self.converter.declared_vars:
            assignment[s] = self.get_value(s)
        return EagerModel(assignment=assignment, environment=self.environment)

    def get_value(self, item):
        self._assert_no_function_type(item)
        sort = item.get_type()
        item = self.converter.convert(item)
        val = self.solver.get_value(item)
        if sort.is_array_type():
            raise NotImplementedError
        elif sort.is_bool_type():
            return self.mgr.Bool(bool(val))
        elif sort.is_bv_type():
            return self.mgr.BV(int(val), sort.width)
        elif sort.is_function_type():
            raise NotImplementedError
        elif sort.is_int_type():
            return self.mgr.Int(int(val), sort.width)
        elif sort.is_real_type():
            raise NotImplementedError
        else:
            raise ConvertExpressionError(f'Unsupported sort: {sort}')


    @clear_pending_pop
    def _reset_assertions(self):
        self.solver.reset()
        self.options(self)

    @clear_pending_pop
    def _add_assertion(self, formula, named=None):
        self._assert_is_boolean(formula)
        term = self.converter.convert(formula)
        self.solver.assert_formula(term)

    @clear_pending_pop
    def _solve(self, assumptions=None):
        if assumptions is None:
            res = self.solver.check_sat()
        else:
            assumptions = [self.converter.convert(a) for a in assumptions]
            res = self.solver.check_sat_assuming(assumptions)

        if res.is_sat():
            return True
        elif res.is_unsat():
            return False
        else:
            raise SolverReturnedUnknownResultError

    @clear_pending_pop
    def _push(self, levels=1):
        self.solver.push(levels)

    @clear_pending_pop
    def _pop(self, levels=1):
        self.solver.pop(levels)


    def _exit(self):
        pass


def check_args(cmp, n):
    def wrapper(f):
        @ft.wraps(f)
        def walk_op(self, formula, args, **kwargs):
            if not cmp(len(args), n):
                raise ConvertExpressionError('Incorrect number of arguments')
            return f(self, formula, args, **kwargs)
        return walk_op
    return wrapper

def make_walk_nary(n, primop):
    @check_args(operator.eq, n)
    def walk_op(self, formula, args, **kwargs):
        res = self.make_term(primop, *args)
        return res
    return walk_op

make_walk_unary = ft.partial(make_walk_nary, 1)
make_walk_binary = ft.partial(make_walk_nary, 2)

def make_walk_variadic(n, primop):
    @check_args(operator.ge, n)
    def walk_op(self, formula, args, **kwargs):
        builder = ft.partial(self.make_term, primop)
        res = ft.reduce(builder, args)
        return res
    return walk_op


class SwitchConverter(Converter, DagWalker):
    def __init__(self, environment, solver):
        DagWalker.__init__(self, environment)
        self.solver = solver
        self.make_term = solver.make_term
        self.make_symbol = solver.make_symbol
        self.make_sort = solver.make_sort
        self.declared_vars = {}
        self.declared_sorts = {}

    @catch_conversion_error
    def convert(self, formula):
        return self.walk(formula)

    def back(self, expr):
        raise NotImplementedError

    def _convert_sort(self, sort):
        try:
            return self.declared_sorts[sort]
        except KeyError:
            pass

        if sort.is_array_type():
            c_sort = self.make_sort(
                ss.sortkinds.ARRAY,
                self._convert_sort(sort.index_type),
                self._convert_sort(sort.elem_type),
            )
        elif sort.is_bool_type():
            c_sort = self.make_sort(ss.sortkinds.BOOL)
        elif sort.is_bv_type():
            c_sort = self.make_sort(ss.sortkinds.BV, sort.width)
        elif sort.is_function_type():
            sig = [self._convert_sort(s) for s in sort.param_types]
            sig.append(self._convert_sort(sort.return_type))
            c_sort = self.make_sort(ss.sortkinds.FUNCTION, sig)
        elif sort.is_int_type():
            c_sort = self.make_sort(ss.sortkinds.INT)
        elif sort.is_real_type():
            c_sort = self.make_sort(ss.sortkinds.REAL)
        else:
            raise ConvertExpressionError(f'Unsupported sort: {sort}')

        return self.declared_sorts.setdefault(sort, c_sort)


    # Declarations
    @check_args(operator.eq, 0)
    def walk_symbol(self, formula, args, **kwargs):
        try:
            return self.declared_vars[formula]
        except KeyError:
            pass

        sort = self._convert_sort(formula.symbol_type())
        res = self.make_symbol(formula.symbol_name(), sort)
        return self.declared_vars.setdefault(formula, res)


    @check_args(operator.eq, 0)
    def _walk_constant(self, formula, args, **kwargs):
        sort = self._convert_sort(formula.constant_type())
        res = self.make_term(formula.constant_value(), sort)
        return res

    walk_bool_constant = _walk_constant
    walk_bv_constant = _walk_constant
    walk_int_constant = _walk_constant
    walk_real_constant = _walk_constant

    # Bool operators
    walk_and = make_walk_variadic(2, ss.primops.And)
    walk_or = make_walk_variadic(2, ss.primops.Or)
    walk_not = make_walk_unary(ss.primops.Not)
    walk_iff = make_walk_variadic(2, ss.primops.Iff)
    walk_implies = make_walk_binary(ss.primops.Implies)

    # Polymorphic Operators
    walk_ite = make_walk_nary(3, ss.primops.Ite)

    def walk_function(self, formula, args, **kwargs):
        name = formula.function_name()
        f = self.walk_symbol(name, name.args())
        res = self.make_term(ss.primops.Apply, f, *args)
        return res

    # Int / real operatos
    walk_lt = make_walk_binary(ss.primops.Lt)
    walk_le = make_walk_binary(ss.primops.Le)
    walk_equals = make_walk_binary(ss.primops.Equal)
    walk_ge = make_walk_binary(ss.primops.Ge)
    walk_gt = make_walk_binary(ss.primops.Gt)

    walk_plus = make_walk_variadic(2, ss.primops.Plus)
    walk_times = make_walk_variadic(2, ss.primops.Mult)

    walk_minus = make_walk_binary(ss.primops.Minus)
    walk_div = make_walk_binary(ss.primops.Div)
    walk_pow = make_walk_binary(ss.primops.Pow)

    # BV Operators
    walk_bv_add = make_walk_binary(ss.primops.BVAdd)
    walk_bv_and = make_walk_binary(ss.primops.BVAnd)
    walk_bv_ashr = make_walk_binary(ss.primops.BVAshr)
    walk_bv_comp = make_walk_binary(ss.primops.BVComp)
    walk_bv_concat = make_walk_binary(ss.primops.Concat)

    @check_args(operator.eq, 1)
    def walk_bv_extract(self, formula, args, **kwargs):
        res = self.make_term(
            ss.Op(
                ss.primops.Extract,
                formula.bv_extract_end(),
                formula.bv_extract_start(),
            ),
            *args
        )
        return res

    walk_bv_lshl = make_walk_binary(ss.primops.BVShl)
    walk_bv_lshr = make_walk_binary(ss.primops.BVLshr)
    walk_bv_mul = make_walk_binary(ss.primops.BVMul)
    walk_bv_neg = make_walk_unary(ss.primops.BVNeg)
    walk_bv_not = make_walk_unary(ss.primops.BVNot)
    walk_bv_or = make_walk_binary(ss.primops.BVOr)

    @check_args(operator.eq, 1)
    def walk_bv_rol(self, formula, args, **kwargs):
        res = self.make_term(
            ss.Op(ss.primops.Rotate_Left, formula.bv_rotation_step()),
            *args
        )
        return res

    @check_args(operator.eq, 1)
    def walk_bv_ror(self, formula, args, **kwargs):
        res = self.make_term(
            ss.Op(ss.primops.Rotate_Right, formula.bv_rotation_step()),
            *args
        )
        return res

    walk_bv_sdiv = make_walk_binary(ss.primops.BVSdiv)

    @check_args(operator.eq, 1)
    def walk_bv_sext(self, formula, args, **kwargs):
        res = self.make_term(
            ss.Op(ss.primops.Sign_Extend, formula.bv_extend_step()),
            *args
        )
        return res

    walk_bv_sle = make_walk_binary(ss.primops.BVSle)
    walk_bv_slt = make_walk_binary(ss.primops.BVSlt)
    walk_bv_srem = make_walk_binary(ss.primops.BVSrem)
    walk_bv_sub = make_walk_binary(ss.primops.BVSub)
    walk_bv_tonatural = make_walk_unary(ss.primops.BV_To_Nat)
    walk_bv_udiv = make_walk_binary(ss.primops.BVUdiv)
    walk_bv_ule = make_walk_binary(ss.primops.BVUle)
    walk_bv_ult = make_walk_binary(ss.primops.BVUlt)
    walk_bv_urem = make_walk_binary(ss.primops.BVUrem)
    walk_bv_xor = make_walk_binary(ss.primops.Xor)

    @check_args(operator.eq, 1)
    def walk_bv_zext(self, formula, args, **kwargs):
        res = self.make_term(
            ss.Op(ss.primops.Zero_Extend, formula.bv_extend_step()),
            *args
        )
        return res

    #array operators
    walk_array_select = make_walk_binary(ss.primops.Select)
    walk_array_store = make_walk_nary(3, ss.primops.Store)


