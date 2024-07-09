from collections import deque

class BinaryNode:
    
    def __init__(self, value):
        assert value is not None, "Value must not be None!"
        
        self.value = value
        self.left = None
        self.right = None
        self.parent = None

class BinarySearchTree:

    def __init__(self):
        self.root = None

    def find(self, value):
        return self._find(self.root, value)

    def _find(self, current, value):
        if current is None:
            return None
        if value == current.value:
            return current
        elif value < current.value:
            return self._find(current.left, value)
        else:
            return self._find(current.right, value)

    def insert_multiple(self, values: list):
        for value in values:
            self.insert(value)

    def insert(self, value):
        new_node = BinaryNode(value)
        if self.root is None:
            self.root = new_node
        else:
            self._insert(self.root, new_node)

    def _insert(self, current_node: BinaryNode, new_node: BinaryNode):
        if new_node.value < current_node.value:
            if current_node.left is None:
                current_node.left = new_node
                new_node.parent = current_node
            else:
                self._insert(current_node.left, new_node)
        else:
            if current_node.right is None:
                current_node.right = new_node
                new_node.parent = current_node
            else:
                self._insert(current_node.right, new_node)

    def _delete(self, node):
        # Case 1: Node has no children
        if node.left is None and node.right is None:
            if node.parent is None:
                self.root = None
            elif node == node.parent.left:
                node.parent.left = None
            else:
                node.parent.right = None

        # Case 2: Node has one child
        elif node.left is None or node.right is None:
            if node.left is not None:
                child = node.left
            else:
                child = node.right

            if node.parent is None:
                self.root = child
            elif node == node.parent.left:
                node.parent.left = child
            else:
                node.parent.right = child

            child.parent = node.parent

        # Case 3: Node has two children
        else:
            successor = self._find_min(node.right)
            node.value = successor.value
            self._delete(successor)

    def _find_min(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current

    def print_tree(self):
        self._print_tree(self.root)

    def _print_tree(self, node):
        if node is not None:
            self._print_tree(node.left)
            print(node.value, end=" ")
            self._print_tree(node.right)

    def print_level_order(self):
        if not self.root:
            return

        queue = deque([self.root])
        
        while queue:
            current = queue.popleft()
            if current:
                print(current.value, end=" ")
                queue.append(current.left)
                queue.append(current.right)
            else:
                print("None", end=" ")

        print()

def main():
    bst1 = BinarySearchTree()

    # Insert multiple values to create a 3-stage tree with some null nodes
    values = [15, 10, 20, None, 12, 17, None]
    for value in values:
        if value is not None:
            bst1.insert(value)

    # Print the tree in level order
    print("Level-order traversal of the tree:")
    bst1.print_level_order()
    
    bst1.de

if __name__ == "__main__":
    main()
