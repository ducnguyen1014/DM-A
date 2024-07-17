from enum import Enum


class Strategy(Enum):
    RED_UNCLE = 0
    LEFT_TRIANGLE = 1
    RIGHT_TRIANGLE = 2
    LEFT_LINE = 3
    RIGHT_LINE = 4


class Color(Enum):
    RED = "R"
    BLACK = "B"


class BinaryNode:

    def __init__(self, value):
        assert value is not None, "Value must not be None!"

        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.color = Color.RED

    def reColor(self):
        self.color = Color.BLACK if self.color == Color.RED else Color.RED


class RedBlackTree:

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
            self.root.color = Color.BLACK
        else:
            self.__insert(self.root, newNode)
        self.__balanceTree(newNode)

    def insertMultiple(self, values: list):
        for value in values:
            self.insert(value)

    # Private insert method
    def __insert(self, currentNode: BinaryNode, newNode: BinaryNode):
        if newNode.value < currentNode.value:
            if currentNode.left is None:
                currentNode.left = newNode
                newNode.parent = currentNode
            else:
                self.__insert(currentNode.left, newNode)
        elif newNode.value > currentNode.value:
            if currentNode.right is None:
                currentNode.right = newNode
                newNode.parent = currentNode
            else:
                self.__insert(currentNode.right, newNode)
        else:
            return

    # Public delete method
    def delete(self, value):
        nodeToDelete = self.__find(self.root, value)
        if nodeToDelete:
            self.__delete(nodeToDelete)

    # Private delete method
    def __delete(self, currentNode: BinaryNode):
        # Case 1: current node has 2 children
        if currentNode.left and currentNode.right:
            successorNode = self.__findMin(currentNode.right)
            currentNode.value = successorNode.value
            currentNode = successorNode
            self.__transplant(successorNode)

        # Case 2: current node has 0 or 1 child
        else:
            self.__transplant(currentNode)

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
            node_repr = "{} ({})".format(str(root.value), str(root.color.value))

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

    # ----------------------------------------------------------------
    # Red black tree methods

    def __isNullOrBlack(self, currentNode: BinaryNode):
        return currentNode == None or currentNode.color == Color.BLACK

    def __getUncleColor(self, currentNode: BinaryNode):
        if (
            currentNode == None
            or currentNode.parent == None
            or currentNode.parent.parent == None
        ):
            return None

        #      Left Uncle
        #
        #                  Grandparent
        #                   /       \
        #                 Uncle     Parent
        #                             \
        #                         Current_Node
        #
        #      Right Uncle
        #
        #                  Grandparent
        #                   /       \
        #                Parent    Uncle
        #                  /
        #            Current_Node

        # Get the color of the Left Uncle
        if currentNode.parent.parent.left == currentNode.parent:
            if self.__isNullOrBlack(currentNode.parent.parent.right):
                return Color.BLACK

            return Color.RED

        elif currentNode.parent.parent.right == currentNode.parent:
            if self.__isNullOrBlack(currentNode.parent.parent.left):
                return Color.BLACK

            return Color.RED

        return Color.RED

    def __rotateLeft(self, currentNode: BinaryNode):

        #          Parent                           Parent
        #            |                                |
        #         DownNode (node)                   UpNode
        #          /     \          =>              /   \
        #        ...    UpNode               DownNode    ...
        #                /  \                 /   \
        #           SubNode?  ...           ...   SubNode?
        #
        #     SubNode maybe NULL

        if currentNode != None and currentNode.right != None:
            parentNode = currentNode.parent
            downNode = currentNode
            upNode = currentNode.right
            subNode = currentNode.right.left

        if subNode:
            subNode.parent = downNode

        upNode.left = downNode
        downNode.right = subNode
        downNode.parent = upNode

        if parentNode:
            upNode.parent = parentNode

            if downNode == parentNode.left:
                parentNode.left = upNode
            else:
                parentNode.right = upNode

        else:
            upNode.parent = None
            self.root = upNode

        return downNode

    def __rotateRight(self, currentNode: BinaryNode):

        #  RotateRight: Maximum 6 Operations since each line contains 2 relationships.
        #
        #         Parent                           Parent
        #            |                                |
        #         DownNode (node)                   UpNode
        #          /     \          =>              /   \
        #      UpNode    ...                      ...   DownNode
        #       /  \                                     /   \
        #    ...   SubNode?                        SubNode?  ...
        #
        #     SubNode maybe NULL

        if currentNode != None and currentNode.left != None:
            parentNode = currentNode.parent
            downNode = currentNode
            upNode = currentNode.left
            subNode = currentNode.left.right

        if subNode:
            subNode.parent = downNode

        upNode.right = downNode
        downNode.left = subNode
        downNode.parent = upNode

        if parentNode:
            upNode.parent = parentNode

            if downNode == parentNode.right:
                parentNode.right = upNode
            else:
                parentNode.left = upNode

        else:
            upNode.parent = None
            self.root = upNode

        return downNode

    # Plug single child to parent of the current node, delete current node
    def __transplant(self, currentNode: BinaryNode):
        # Case 1: Current node is the only node of the tree
        if not currentNode.parent and not currentNode.left and not currentNode.right:
            self.root = None

        # Case 2: Current node is a leaf node
        if currentNode.left and currentNode.right:
            # Left leaf node
            if currentNode.parent.left == currentNode:
                # If the current node is red, just delete
                currentNode.parent.left = None

                # If the current node is black => Balance
                if currentNode.color == Color.BLACK:
                    currentNode = self.__rotateLeft(currentNode.parent)
                    if currentNode.right:
                        self.__balanceTree(currentNode.right)

            # Rigth leaf node
            if currentNode.parent.right == currentNode:
                # If the current node is red, just delete
                currentNode.parent.right = None

                # If the current node is black => Balance
                currentNode = self.__rotateRight(currentNode.parent)
                if currentNode.left:
                    self.__balanceTree(currentNode.left)

        # Case 3: Current node has 1 child
        replacementNode = currentNode.left if currentNode.left else currentNode.right

        if not currentNode.parent:
            self.root = replacementNode
        elif currentNode.parent.left == currentNode:
            currentNode.parent.left = replacementNode
        else:
            currentNode.parent.right = replacementNode

        if replacementNode:
            replacementNode.parent = currentNode.parent
            replacementNode.color = Color.BLACK

    def __getStrategy(self, currentNode: BinaryNode):
        # Not valid
        if (
            currentNode == None
            or currentNode.parent == None
            or currentNode.parent.parent == None
        ):
            return None

        # RED UNCLE
        elif self.__getUncleColor(currentNode) == Color.RED:
            return Strategy.RED_UNCLE

        # LEFT_TRIANGLE
        elif (
            currentNode.parent == currentNode.parent.parent.left
            and currentNode == currentNode.parent.right
        ):
            return Strategy.LEFT_TRIANGLE

        # RIGHT_TRIANGLE
        elif (
            currentNode.parent == currentNode.parent.parent.right
            and currentNode == currentNode.parent.left
        ):
            return Strategy.RIGHT_TRIANGLE

        # LEFT LINE
        elif (
            currentNode.parent == currentNode.parent.parent.left
            and currentNode == currentNode.parent.left
        ):
            return Strategy.LEFT_LINE

        # RIGHT_LINE
        elif (
            currentNode.parent == currentNode.parent.parent.right
            and currentNode == currentNode.parent.right
        ):
            return Strategy.RIGHT_LINE

    def __balanceTree(self, currentNode: BinaryNode):
        if (
            currentNode == None
            or currentNode.parent == None
            or currentNode.parent.parent == None
        ):
            return

        # ROOT IS RED
        if currentNode == self.root and currentNode.color == Color.RED:
            currentNode.reColor()

        while (
            currentNode.parent != None
            and currentNode.color == Color.RED
            and currentNode.parent.color == Color.RED
        ):
            # Get strategy
            currentStrategy = self.__getStrategy(currentNode)

            # RED UNCLE
            if currentStrategy == Strategy.RED_UNCLE:
                currentNode.parent.parent.left.reColor()
                currentNode.parent.parent.right.reColor()

                if currentNode.parent.parent != self.root:
                    currentNode.parent.parent.reColor()

                currentNode = currentNode.parent.parent

            # LEFT TRIANGLE
            elif currentStrategy == Strategy.LEFT_TRIANGLE:
                currentNode = currentNode.parent
                currentNode = self.__rotateLeft(currentNode)

            # RIGHT TRIANGLE
            elif currentStrategy == Strategy.RIGHT_TRIANGLE:
                currentNode = currentNode.parent
                currentNode = self.__rotateRight(currentNode)

            # LEFT LINE
            elif currentStrategy == Strategy.LEFT_LINE:
                currentNode = currentNode.parent.parent
                currentNode = self.__rotateRight(currentNode)
                currentNode.reColor()
                currentNode.parent.reColor()

            # RIGHT LINE
            elif currentStrategy == Strategy.RIGHT_LINE:
                currentNode = currentNode.parent.parent
                currentNode = self.__rotateLeft(currentNode)
                currentNode.reColor()
                currentNode.parent.reColor()


def main():
    bst1 = RedBlackTree()

    # Insert multiple values to create a 3-stage tree with some null nodes
    values = [11, 15, 5, 18, 21, 24, 27, 29]
    for value in values:
        if value is not None:
            bst1.insert(value)

    bst1.printTree()

    bst1.delete(24)

    bst1.printTree()


if __name__ == "__main__":
    main()
