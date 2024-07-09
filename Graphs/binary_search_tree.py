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

    # Public find method
    def find(self, value):
        return self.__find(self.root, value) is not None

    # Private find method
    def __find(self, current, value):
        if current is None:
            return None
        if value == current.value:
            return current
        elif value < current.value:
            return self.__find(current.left, value)
        else:
            return self.__find(current.right, value)

    # Public insert method
    def insert(self, value):
        newNode = BinaryNode(value)
        if self.root is None:
            self.root = newNode
        else:
            self._insert(self.root, newNode)

    def insertMultiple(self, values: list):
        for value in values:
            self.insert(value)

    # Private insert method
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

    # Public delete method
    def delete(self, value):
        if self.root is None:
            return

        nodeToDelete = self.__find(self.root, value)
        if nodeToDelete:
            self.__delete(nodeToDelete)

    # Private delete method
    def __delete(self, node):
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
            child = node.left if node.left else node.right

            if node.parent is None:
                self.root = child
            elif node == node.parent.left:
                node.parent.left = child
            else:
                node.parent.right = child

            child.parent = node.parent

        # Case 3: Node has two children
        else:
            successor = self.__findMin(node.right)
            node.value = successor.value
            self.__delete(successor)

    def __findMin(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current

    def printTree(self):
        lines = self.__buildTreeString(self.root, 0, False, "-")[0]
        print("\n" + "\n".join((line.rstrip() for line in lines)))

    def __buildTreeString(self, root, curr_index, index=False, delimiter="-"):
        if root is None:
            return [], 0, 0, 0

        line1 = []
        line2 = []
        if index:
            node_repr = "{}{}{}".format(curr_index, delimiter, root.value)
        else:
            node_repr = str(root.value)

        new_root_width = gap_size = len(node_repr)

        # Get the left and right sub-boxes, their widths, and root repr positions
        l_box, l_box_width, l_root_start, l_root_end = self.__buildTreeString(
            root.left, 2 * curr_index + 1, index, delimiter
        )
        r_box, r_box_width, r_root_start, r_root_end = self.__buildTreeString(
            root.right, 2 * curr_index + 2, index, delimiter
        )

        # Draw the branch connecting the current root node to the left sub-box
        # Pad the line with whitespaces where necessary
        if l_box_width > 0:
            l_root = (l_root_start + l_root_end) // 2 + 1
            line1.append(" " * (l_root + 1))
            line1.append("_" * (l_box_width - l_root))
            line2.append(" " * l_root + "/")
            line2.append(" " * (l_box_width - l_root))
            new_root_start = l_box_width + 1
            gap_size += 1
        else:
            new_root_start = 0

        # Draw the representation of the current root node
        line1.append(node_repr)
        line2.append(" " * new_root_width)

        # Draw the branch connecting the current root node to the right sub-box
        # Pad the line with whitespaces where necessary
        if r_box_width > 0:
            r_root = (r_root_start + r_root_end) // 2
            line1.append("_" * r_root)
            line1.append(" " * (r_box_width - r_root + 1))
            line2.append(" " * r_root + "\\")
            line2.append(" " * (r_box_width - r_root))
            gap_size += 1
        new_root_end = new_root_start + new_root_width - 1

        # Combine the left and right sub-boxes with the branches drawn above
        gap = " " * gap_size
        new_box = ["".join(line1), "".join(line2)]
        for i in range(max(len(l_box), len(r_box))):
            l_line = l_box[i] if i < len(l_box) else " " * l_box_width
            r_line = r_box[i] if i < len(r_box) else " " * r_box_width
            new_box.append(l_line + gap + r_line)

        # Return the new box, its width and its root repr positions
        return new_box, len(new_box[0]), new_root_start, new_root_end


def main():
    bst1 = BinarySearchTree()

    # Insert multiple values to create a 3-stage tree with some null nodes
    values = [35, 92, 15, 71, 32, 39, 65, 88, 8, 61, 100, 102]
    for value in values:
        if value is not None:
            bst1.insert(value)

    # Print the tree structure
    print("Tree structure:")
    bst1.printTree()

    # Find
    print("Finding 65:", bst1.find(65))

    # Delete a node and print the tree again
    bst1.delete(65)

    print("Tree structure after deleting 10:")
    bst1.printTree()

    # Find
    print("Finding 65:", bst1.find(65))


if __name__ == "__main__":
    main()
