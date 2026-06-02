# # # from minizinc import Instance, Model, Solver

# # # gecode = Solver.lookup("gecode")

# # # model = Model()
# # # model.add_string(
# # #     """
# # #     enum DAY = {Mo, Tu, We, Th, Fr};
# # #     var DAY: d;
# # #     constraint d = max(DAY);
# # #     """
# # # )
# # # instance = Instance(gecode, model)

# # # result = instance.solve()
# # # print(result["d"])  # Mo
# # # assert isinstance(result["d"], str)

# # from minizinc import Instance, Model, Solver

# # gecode = Solver.lookup("gecode")

# # model = Model()
# # model.add_string(
# #     """
# #     include "all_different.mzn";
# #     set of int: A;
# #     set of int: B;
# #     array[A] of var B: arr;
# #     var set of B: X;
# #     var set of B: Y;

# #     constraint all_different(arr);
# #     constraint forall (i in index_set(arr)) ( arr[i] in X );
# #     constraint forall (i in index_set(arr)) ( (arr[i] mod 2 = 0) <-> arr[i] in Y );
# #     """
# # )

# # instance = Instance(gecode, model)
# # instance["A"] = range(3, 8)  # MiniZinc: 3..7
# # instance["B"] = {4, 3, 2, 1, 0}  # MiniZinc: {4, 3, 2, 1, 0}

# # result = instance.solve()
# # print(result["X"])  # {0, 1, 2, 3, 4}
# # assert isinstance(result["X"], set)
# # print(result["Y"])  # {0, 2, 4}
# # assert isinstance(result["Y"], set)

# from minizinc import Instance, Model, Solver

# gecode = Solver.lookup("gecode")

# model = Model()
# model.add_string(
#     """
#     include "all_different.mzn";
#     array[1..4] of var 1..10: x;
#     constraint all_different(x);
#     """
# )
# instance = Instance(gecode, model)

# with instance.branch() as opt:
#     opt.add_string("solve maximize sum(x);\n")
#     res = opt.solve()
#     obj = res["objective"]

# instance.add_string(f"constraint sum(x) = {obj};\n")

# result = instance.solve(all_solutions=True)
# for sol in result.solution:
#     print(sol.x)

from dataclasses import InitVar, dataclass
from typing import List

from minizinc import Instance, Model, Solver


@dataclass
class TaskAssignment:
    task: List[int]
    objective: int
    __output_item: InitVar[str] = None

    def check(self) -> bool:
        return len(set(self.task)) == len(self.task)


gecode = Solver.lookup("gecode")
model = Model()
model.add_string(
    """
    int: n;
    set of int: DOM = 1..n;
    int: m;
    set of int: COD = 1..m;
    array[DOM,COD] of int: profit;

    array[DOM] of var COD: task;

    include "all_different.mzn";
    constraint all_different(task);

    solve maximize sum(w in DOM)
                (profit[w,task[w]]);
    """
)
model.output_type = TaskAssignment

inst = Instance(gecode, model)
inst["n"] = 4
inst["m"] = 5
inst["profit"] = [[7, 1, 3, 4, 6], [8, 2, 5, 1, 4], [4, 3, 7, 2, 5], [3, 1, 6, 3, 6]]


sol = inst.solve().solution
assert type(sol) == TaskAssignment

if sol.check:
    print("A valid assignment!")
else:
    print("A bad assignment!")