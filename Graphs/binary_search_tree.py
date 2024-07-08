class BinaryNode:

    def __init__(self, value):
        assert value != None, "Value must not be None!"

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

    def insertMutiple(self, values: list):
        for value in values:
            self.insert(value)

    def insert(self, value):
        newNode = BinaryNode(value)
        if self.root is None:
            self.root = newNode
        else:
            self._insert(self.root, newNode)

    def _insert(self, currentNode: BinaryNode, newNode: BinaryNode):
        if newNode.value < currentNode.value:
            if currentNode.left is None:
                currentNode.left = newNode
                newNode.parent = currentNode
            else:
                self._insert(currentNode.left, newNode)
        else:
            if currentNode.right is None:
                currentNode.right = newNode
                newNode.parent = currentNode
            else:
                self._insert(currentNode.right, newNode)

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


def main():
    bst1 = BinarySearchTree()
    bst1.insert(10)
    bst1.insert(20)

