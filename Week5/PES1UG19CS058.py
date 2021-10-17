import numpy as np


class Tensor:

    """
    Tensor Wrapper for Numpy arrays.
    Implements some binary operators.
    Array Broadcasting is disabled
    Args:
        arr: Numpy array (numerical (int, float))
        requires_grad: If the tensor requires_grad (bool)(otherwise gradient dont apply to the tensor)
    """

    def __init__(self, arr, requires_grad=True):

        self.arr = arr
        self.requires_grad = requires_grad

        # When node is created without predecessor the op is denoted as 'leaf'
        # 'leaf' signifies leaf node
        self.history = ['leaf', None, None]
        # History stores the information of the operation that created the Tensor.
        # Check set_history

        # Gradient of the tensor
        self.zero_grad()
        self.shape = self.arr.shape

    def zero_grad(self):
        """
        Set grad to zero
        """
        self.grad = np.zeros_like(self.arr)

    def set_history(self, op, operand1, operand2):
        """
        Set History of the node, indicating how the node was created.
        Ex:-
            history -> ['add', operand1(tensor), operand2(tensor)]
            history -> ['leaf', None, None] if tensor created directly
        Args:
            op: {'add', 'sub', 'mul', 'pow', 'matmul', 'leaf') (str)
            operand1: First operand to the operator. (Tensor object)
            operand2: Second operand to the operator. (Tensor object)
        """
        self.history = []
        self.history.append(op)
        self.requires_grad = False
        self.history.append(operand1)
        self.history.append(operand2)

        if operand1.requires_grad or operand2.requires_grad:
            self.requires_grad = True

    """
    Addition Operation
    Tensor-Tensor(Element Wise)
    __add__: Invoked when left operand of + is Tensor
    grad_add: Gradient computation through the add operation
    """

    def __add__(self, other):
        """
        Args:
            other: The second operand.(Tensor)
                    Ex: a+b then other -> b, self -> a
        Returns:
            Tensor: That contains the result of operation
        """
        if isinstance(other, self.__class__):
            if self.shape != other.shape:
                raise ArithmeticError(
                    f"Shape mismatch for +: '{self.shape}' and '{other.shape}' ")
            out = self.arr + other.arr
            out_tensor = Tensor(out)
            out_tensor.set_history('add', self, other)

        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{self.__class__}' and '{type(other)}'")

        return out_tensor

    """
    Matrix Multiplication Operation (@)
    Tensor-Tensor
    __matmul__: Invoked when left operand of @ is Tensor
    grad_matmul: Gradient computation through the matrix multiplication operation
    """

    def __matmul__(self, other):
        """
        Args:
            other: The second operand.(Tensor)
                    Ex: a+b then other -> b, self -> a
        Returns:
            Tensor: That contains the result of operation
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"unsupported operand type(s) for matmul: '{self.__class__}' and '{type(other)}'")
        if self.shape[-1] != other.shape[-2]:
            raise ArithmeticError(
                f"Shape mismatch for matmul: '{self.shape}' and '{other.shape}' ")
        out = self.arr @ other.arr
        out_tensor = Tensor(out)
        out_tensor.set_history('matmul', self, other)

        return out_tensor

    def grad_add(self, gradients=None):
        output1 = self.history[1]
        output2 = self.history[2]

        for output in [output1,output2]:
            output.zero_grad()

        if output2.requires_grad and gradients not None: 
            output2.grad += np.ones_like(output2.arr)
        if output1.requires_grad and gradients not None: 
            output1.grad += np.ones_like(output1.arr)
        
        if gradients is None:
            return (output1.grad,output2.grad)

        if output2.requires_grad and gradients not None: 
            output2.grad = np.multiply(gradients,np.ones_like(output2.arr))   
        if output1.requires_grad and gradients not None: 
            output1.grad = np.multiply(gradients,np.ones_like(output1.arr))
        
        return (output1.grad, output2.grad)

    def grad_matmul(self, gradients=None):
        output1 = self.history[1]
        output2 = self.history[2]

        if gradients is None:
        	if output1.requires_grad and gradients not None:
        		output1.grad += np.matmul(np.ones_like(output1.arr), output2.arr.transpose())
        	if output2.requires_grad and gradients not None:
        		output2.grad += (np.matmul(np.ones_like(output2.arr), output1.arr)).transpose()
        else:
        	if output1.requires_grad and gradients not None:
        		output1.grad += np.multiply(gradients,np.matmul(np.ones_like(output1.arr), output2.arr.transpose()))
        	if output2.requires_grad and gradients not None:
        		output2.grad += np.multiply(gradients,np.matmul(np.ones_like(output2.arr), output1.arr).transpose())
     
        return (output1.grad, output2.grad)

    def backward(self, gradients=None):
        if self.requires_grad == None: 
            return
        if self.history[0] == 'add':
            gradient = self.grad_add(gradients)
            if self.history[2]:
                self.history[2].backward(gradient[1])
            if self.history[1]:
                self.history[1].backward(gradient[0])
            
        elif self.history[0] == 'matmul':
            gradient = self.grad_matmul(gradients)
            if self.history[2]:
                self.history[2].backward(gradient[1])
            if self.history[1]:
                self.history[1].backward(gradient[0])
        elif self.requires_grad:
                self.grad = gradients
    
       
       
        		
