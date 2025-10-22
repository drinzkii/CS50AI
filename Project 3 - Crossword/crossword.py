import sys
from collections import deque
from crossword import Crossword, Variable

class CrosswordCreator():
    """
    Crossword puzzle solver using CSP with backtracking and AC-3.
    """

    def __init__(self, crossword: Crossword):
        self.crossword = crossword

        # Domains: mapping variable -> set of possible words
        self.domains = {
            var: set(self.crossword.words)
            for var in self.crossword.variables
        }

    def enforce_node_consistency(self):
        """
        Make each variable node-consistent:
        remove any word from a variable's domain that is not the correct length.
        """
        for var in list(self.domains.keys()):
            for word in set(self.domains[var]):
                if len(word) != var.length:
                    self.domains[var].remove(word)

    def revise(self, x: Variable, y: Variable) -> bool:
        """
        Make variable x arc-consistent with variable y.
        Remove values from self.domains[x] that have no possible compatible value in self.domains[y].

        Return True if a revision was made (i.e., some value was removed), else False.
        """
        revised = False
        overlap = self.crossword.overlaps.get((x, y))
        if overlap is None:
            return False

        i, j = overlap
        to_remove = set()
        for word_x in self.domains[x]:
            # if no word_y in domain[y] matches at overlap character, mark for removal
            has_support = False
            for word_y in self.domains[y]:
                if word_x[i] == word_y[j]:
                    has_support = True
                    break
            if not has_support:
                to_remove.add(word_x)

        if to_remove:
            for w in to_remove:
                self.domains[x].remove(w)
            revised = True

        return revised

    def ac3(self, arcs=None) -> bool:
        """
        AC-3 algorithm. If arcs is None, initialize with all arcs (x,y) where x != y and overlap exists.
        Otherwise, start with provided arcs (list of (x,y) tuples).
        Return False if any domain becomes empty; True otherwise.
        """
        queue = deque()
        if arcs is None:
            # all arcs where there is an overlap
            for x in self.crossword.variables:
                for y in self.crossword.neighbors(x):
                    queue.append((x, y))
        else:
            for arc in arcs:
                queue.append(arc)

        while queue:
            x, y = queue.popleft()
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                # add all neighbors z of x, except y, back to queue
                for z in self.crossword.neighbors(x):
                    if z != y:
                        queue.append((z, x))
        return True

    def assignment_complete(self, assignment: dict) -> bool:
        """Return True if assignment assigns every variable."""
        return set(assignment.keys()) == set(self.crossword.variables)

    def consistent(self, assignment: dict) -> bool:
        """
        Check if assignment is consistent:
        - All assigned words are unique.
        - Each word has correct length.
        - No conflicting letters at overlaps.
        """
        # All values distinct
        if len(set(assignment.values())) != len(assignment):
            return False

        for var, word in assignment.items():
            # correct length
            if len(word) != var.length:
                return False

            # check overlaps with assigned neighbors
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    overlap = self.crossword.overlaps.get((var, neighbor))
                    if overlap is None:
                        continue
                    i, j = overlap
                    if word[i] != assignment[neighbor][j]:
                        return False
        return True

    def order_domain_values(self, var: Variable, assignment: dict):
        """
        Return list of domain values for var ordered by least-constraining value heuristic.
        """
        values = list(self.domains[var])

        def ruled_out_count(value):
            count = 0
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    continue
                overlap = self.crossword.overlaps.get((var, neighbor))
                if overlap is None:
                    continue
                i, j = overlap
                # count neighbor domain values that would be incompatible
                for w in self.domains[neighbor]:
                    if value[i] != w[j]:
                        count += 1
            return count

        values.sort(key=ruled_out_count)
        return values

    def select_unassigned_variable(self, assignment: dict):
        """
        Select an unassigned variable using MRV, then degree heuristic.
        """
        unassigned = [v for v in self.crossword.variables if v not in assignment]

        # MRV: fewest remaining values in domain
        # Degree heuristic tie-breaker: most neighbors
        def sort_key(v):
            return (len(self.domains[v]), -len(self.crossword.neighbors(v)))

        unassigned.sort(key=sort_key)
        return unassigned[0] if unassigned else None

    def backtrack(self, assignment: dict):
        """
        Backtracking search to find a complete assignment. Returns assignment or None.
        """
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(var, assignment):
            # create new assignment
            assignment[var] = value
            if self.consistent(assignment):
                # preserve domains to restore later
                domains_backup = {v: set(self.domains[v]) for v in self.domains}
                # assign var to single value in domains and run AC-3 to propagate constraints
                self.domains[var] = {value}

                # initial arcs: all (neighbor, var)
                arcs = [(neighbor, var) for neighbor in self.crossword.neighbors(var)]
                if self.ac3(arcs):
                    result = self.backtrack(assignment)
                    if result is not None:
                        return result

                # restore domains if failure
                self.domains = domains_backup

            # remove assignment for var and try next value
            assignment.pop(var, None)

        return None

    def solve(self):
        """
        Solve the crossword: enforce node consistency, AC-3, then backtrack.
        Returns a complete assignment or None.
        """
        # Node consistency
        self.enforce_node_consistency()

        # AC-3 initially
        if not self.ac3():
            return None

        # Backtracking search
        return self.backtrack(dict())


def main():
    if len(sys.argv) not in [3, 4]:
        print("Usage: python generate.py structure.txt words.txt [output.png]")
        sys.exit(1)

    structure, words = sys.argv[1], sys.argv[2]
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    solution = creator.solve()

    if solution is None:
        print("No solution.")
    else:
        # Print assignment: variable -> word
        for var, word in solution.items():
            print(f"{var.direction} at ({var.i},{var.j}) length {var.length}: {word}")


if __name__ == "__main__":
    main()
