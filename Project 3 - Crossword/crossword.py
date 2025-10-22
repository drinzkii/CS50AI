import sys

from crossword import *


class CrosswordCreator():
    """Try to solve a crossword by enforcing constraints and using backtracking."""

    def __init__(self, crossword):
        """Create new CSP crossword generate."""

        self.crossword = crossword
        self.domains = {
            var: set(self.crossword.words)
            for var in self.crossword.variables
        }

    def enforce_node_consistency(self):
        """Update `self.domains` such that each variable is node-consistent.

        In this problem, node consistency means removing any words whose
        length is not equal to the variable's length.
        """
        for var in list(self.domains.keys()):
            for word in list(self.domains[var]):
                if len(word) != var.length:
                    self.domains[var].remove(word)

    def revise(self, x, y):
        """Make variable `x` arc consistent with variable `y`.

        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value in `self.domains[y]` that does not
        conflict on their overlap. Return True if a revision was made.
        """
        revised = False
        # Get overlap indices between x and y
        overlap = None
        # overlaps may be keyed by (x, y)
        if hasattr(self.crossword, 'overlaps'):
            overlap = self.crossword.overlaps.get((x, y))
            if overlap is None:
                overlap = self.crossword.overlaps.get((y, x))
                # if we got (y,x) we should flip indices when comparing
                flipped = True if overlap is not None else False
            else:
                flipped = False
        else:
            overlap = None
            flipped = False

        # If there's no overlap, nothing to revise
        if overlap is None:
            return False

        # Determine indices depending on whether we flipped the order
        if not flipped:
            i, j = overlap
        else:
            # if overlap came from (y,x), then indices are reversed
            j, i = overlap

        # For each value in x's domain, check if there exists a value in y's domain
        # that matches on the overlapping character position.
        for word_x in list(self.domains[x]):
            # if there exists some word_y in domains[y] such that word_x[i] == word_y[j]
            satisfied = False
            for word_y in self.domains[y]:
                if word_x[i] == word_y[j]:
                    satisfied = True
                    break
            if not satisfied:
                self.domains[x].remove(word_x)
                revised = True

        return revised

    def ac3(self, arcs=None):
        """Enforce arc consistency in the CSP.

        If `arcs` is None, initialize the queue with all arcs in the problem.
        Otherwise, start with the given list of arcs.
        Return True if arc consistency is enforced without emptying any domain;
        return False if some domain is emptied.
        """
        # Build initial queue
        queue = []
        if arcs is None:
            # all pairs (x,y) where x != y and there is an overlap
            for x in self.crossword.variables:
                for y in self.crossword.neighbors(x):
                    queue.append((x, y))
        else:
            queue = list(arcs)

        while queue:
            x, y = queue.pop(0)
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                for z in self.crossword.neighbors(x):
                    if z != y:
                        queue.append((z, x))
        return True

    def assignment_complete(self, assignment):
        """Return True if `assignment` is complete (i.e., all variables assigned).
        """
        return set(assignment.keys()) == set(self.crossword.variables)

    def consistent(self, assignment):
        """Return True if `assignment` is consistent.

        Checks that all values are distinct, each word matches variable length,
        and no neighboring variables conflict on their overlap.
        """
        # All values must be unique
        values = list(assignment.values())
        if len(values) != len(set(values)):
            return False

        # Each value must be correct length and not conflict with neighbors
        for var, word in assignment.items():
            if len(word) != var.length:
                return False
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    # check overlap
                    overlap = self.crossword.overlaps.get((var, neighbor))
                    flipped = False
                    if overlap is None:
                        overlap = self.crossword.overlaps.get((neighbor, var))
                        if overlap is not None:
                            flipped = True
                    if overlap is None:
                        continue
                    if not flipped:
                        i, j = overlap
                    else:
                        j, i = overlap
                    if word[i] != assignment[neighbor][j]:
                        return False
        return True

    def order_domain_values(self, var, assignment):
        """Return domain values for `var` ordered by the least-constraining value heuristic.

        That is, prefer values that rule out the fewest choices for neighboring
        unassigned variables.
        """
        # For each value in var's domain, count the number of ruled-out values
        counts = []
        for value in self.domains[var]:
            eliminated = 0
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    continue
                overlap = self.crossword.overlaps.get((var, neighbor))
                flipped = False
                if overlap is None:
                    overlap = self.crossword.overlaps.get((neighbor, var))
                    if overlap is not None:
                        flipped = True
                if overlap is None:
                    continue
                if not flipped:
                    i, j = overlap
                else:
                    j, i = overlap
                # count neighbor domain words that would conflict
                for w in self.domains[neighbor]:
                    if value[i] != w[j]:
                        eliminated += 1
            counts.append((value, eliminated))

        # Sort by eliminated ascending (least constraining first)
        ordered = sorted(counts, key=lambda x: x[1])
        return [v for v, _ in ordered]

    def select_unassigned_variable(self, assignment):
        """Return an unassigned variable not in `assignment`, chosen by MRV then degree heuristic."""
        unassigned = [v for v in self.crossword.variables if v not in assignment]
        # Minimum Remaining Values heuristic: fewest domain values
        # Tie-breaker: largest degree (most neighbors)
        def sort_key(v):
            return (len(self.domains[v]), -len(self.crossword.neighbors(v)))

        unassigned.sort(key=sort_key)
        return unassigned[0]

    def backtrack(self, assignment):
        """Use backtracking search to find a complete assignment.

        Returns a complete assignment dictionary or None if failure.
        """
        # If assignment is complete, return it
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = value

            if self.consistent(new_assignment):
                # Inference: temporarily reduce domain and enforce AC3
                saved_domains = {v: set(self.domains[v]) for v in self.domains}
                self.domains[var] = {value}

                # Prepare arcs for AC3: all arcs (neighbor, var)
                arcs = [(neighbor, var) for neighbor in self.crossword.neighbors(var)]
                if self.ac3(arcs):
                    result = self.backtrack(new_assignment)
                    if result is not None:
                        return result

                # restore domains
                self.domains = {v: set(saved_domains[v]) for v in saved_domains}

        return None


def main():
    # Parse command-line arguments
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)

    creator.enforce_node_consistency()
    creator.ac3()

    assignment = creator.backtrack(dict())

    if assignment is None:
        print("No solution.")
    else:
        print(assignment)
        if output:
            creator.crossword.print(assignment)


if __name__ == "__main__":
    main()
