> 题目来源：[LeetCode Hot 100](https://leetcode-cn.com/problemset/leetcode-hot-100/)
>
> 题解来源：个人解法和LeetCode题解(没有对他人的题解注明，抱歉)

# 78.子集
[题目](https://leetcode-cn.com/problems/subsets/)

## 迭代法

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans=[[]]
        for x in nums:
            ans+=[[x]+i for i in ans]
        return ans
```
对于`nums=[1,2,3]`
图示如下：

``` 
[]
[] [1]
[] [1] [2] [1,2]
[] [1] [2] [1,2] [3] [1,3] [2,3] [1,2,3]
```

## 库函数法

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans=[]
        for i in range(len(nums)+1):
            for x in itertools.combinations(nums, i):
                ans.append(x)
        return ans
```

`itertools.combinations(i,n)`函数返回一个迭代器，包含序列i中长度为n的子序列

## *回溯算法*

```python
class Solution:
    def subsets(self, nums: List[int])-> List[List[int]]:
        ans=[]
        n=len(nums)
        
        def helper(i:int,temp:List[int]):
            ans.append(temp)
            for j in range(i,n):
                helper(j+1,temp+[nums[j]])
        helper(0,[])
        return ans
```

# 617. 合并二叉树
[题目](https://leetcode-cn.com/problems/merge-two-binary-trees/)

## 递归

```python
class Solution:
    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
        if not t1 or not t2:
            return t1 or t2
        t1.val=t1.val+t2.val
        t1.left=self.mergeTrees(t1.left,t2.left)
        t1.right=self.mergeTrees(t1.right,t2.right)
        return t1
```

如果两个节点都不空，递归调用即可；如果有空的，则返回非空的那个（都空返回None）

如果不改变原树的元素，可以这样写：

```python
class Solution:
    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
        if t1==None and t2==None:
            return None;
        else:
            x=t1.val if t1 else 0
            y=t2.val if t2 else 0
            return_node=TreeNode(x+y)
            return_node.left=self.mergeTrees(t1 and t1.left,t2 and t2.left)
            return_node.right=self.mergeTrees(t1 and t1.right,t2 and t2.right)
        return return_node   
```

# 461. 汉明距离
[题目](https://leetcode-cn.com/problems/hamming-distance/)

## 异或

```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
    	return bin(x^y).count("1")
```

异或，相同的位为0，不同的位为1；`bin()`把int型转为二进制，用字符串表示

## 移位法

同样利用异或，但是这里通过移位法计算1的个数

```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        xor=x^y
        ans=0
        while xor:
            if xor & 1:
                ans+=1
            xor>>=1
        return ans
```



# 226. 翻转二叉树
[题目](https://leetcode-cn.com/problems/invert-binary-tree/)

## 递归法

```python
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if  root==None or (root.right==None and root.left==None):
            return root
        root.left,root.right=self.invertTree(root.right),self.invertTree(root.left)
        return root
```

节点为空或者没有子节点，直接返回该节点（因为不需要翻转子节点）；若有子节点，则对子节点分别翻转，之后交换两个节点即可（利用解包操作）

# 46. 全排列
[题目](https://leetcode-cn.com/problems/permutations/)

## 库函数法

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
    	return list(itertools.permutations(nums))
```

`itertools.permutations(i)`返回可迭代对象i的全排列

## 回溯法

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def full_permutation(nums: List[int],start: int=0):
            if start==len(nums)-1:
                ans.append(nums.copy())
            else:
                i=start
                while i<len(nums):
                    nums[start],nums[i]=nums[i],nums[start]
                    full_permutation(nums,start+1)
                    nums[start],nums[i]=nums[i],nums[start]
                    i+=1
                    
        if nums==None:
            return None
        ans=list()
        full_permutation(nums)
        return ans 
```

`while`循环是核心；全排列就是从第一个数字起每个数分别与它后面的数字交换



# 22. 括号生成
[题目](https://leetcode-cn.com/problems/generate-parentheses/)

## 回溯法

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res=list()
        def backtrack(left:int,right:int,s:str):
            if left>n or left<right:
                return
            elif (left+right)==2*n:
                res.append(s)
                return
            backtrack(left+1,right,s+'(')
            backtrack(left,right+1,s+')')
            return
        backtrack(0,0,"")
        return res
```

易知最左边的括号必是`(`。从这里开始逐步增加`(或)`，增加后有以下情况：

* 先考虑无效情况，

    * 左括号数大于n（即一半括号数），这种情况肯定不行
    * 左括号数小于右括号数，无法达成有效，也不行

* 排除以上无效情况后，剩下的情况都是继续增加括号也可能有效的：

    * 如果两括号数目和为2n，则为结果
    * 如果数目还不够，则继续增加括号，这回产生加左和右两种结果的递归调用

    

    其递归调用结构如下：
    
    ![](./images/lc22.png)

# 338. 比特位计数
[题目](https://leetcode-cn.com/problems/counting-bits/)

## 库函数法

```python
class Solution:
    def countBits(self, num: int) -> List[int]:
        return [(bin(x)).count("1") for x in range(num+1)]
```

类似[461. 汉明距离](https://leetcode-cn.com/problems/hamming-distance/)，`bin()`函数返回一个数的二进制（字符串形式），在用`count()`方法计算1的个数即可；最后用列表构造式构造List即可

## 动态规划

二进制中1的个数规律如下：
$$
n(i)=\begin{cases}
n(i-1)+1 \ \ i为奇数 \\
n(\frac{i}{2}) \ \  i为偶数
\end{cases}
$$

```python
class Solution:
    def countBits(self, num: int) -> List[int]:
        if num == 0:
            return [0]
        res=[0]
        for i in range(1,num+1):
            if i%2==0:
                res.append(res[i//2])
            else:
                res.append(res[i-1]+1)
        return res
```



# 104. 二叉树的最大深度
[题目](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

## 递归

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if root==None:
            return 0
        else:
            return 1+max(self.maxDepth(root.left),self.maxDepth(root.right))
```

若节点为空，则深度为1；不空，则深度为深度最大的子节点的深度加1

## 广度优先搜索

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        q=deque([root])
        d=0
        while q:
            size=len(q)
            while size>0:
                node=q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
                size-=1
            d+=1
        return d 
```

建立一个队列，根节点入队；每次出队一层，同时该层的子节点入队；队列为空时即可得层数



# 94. 二叉树的中序遍历
[题目](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

## 递归

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res=list()
        def tra(root:TreeNode):
            if root==None:
                return
            else:
                tra(root.left)
                res.append(root.val)
                tra(root.right)
        tra(root)
        return res
```

## 栈模拟

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        stack=list()
        res=list()
        node=root
        while node or stack:
            while node:
                stack.append(node)
                node=node.left
            node=stack.pop()
            res.append(node.val)
            node=node.right
        return res
```

创建一个栈：

* 根节点入栈，之后如果其左子节点存在，则每次都把左子节点入栈，直至没有左子节点后
* 弹出栈顶节点，记录，再把该节点的右子节点入栈，重复上述操作，直至栈和当前节点都为空



## 标记+栈模拟

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        WHITE,GRAY = 0, 1
        res=[]
        stack=[(WHITE, root)]
        while stack:
            color,node=stack.pop()
            if not node: 
                continue
            if color == WHITE:
                stack.append((WHITE, node.right))
                stack.append((GRAY, node))
                stack.append((WHITE, node.left))
            else:
                res.append(node.val)
        return res
    
    
    class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        WHITE, GRAY = 0, 1
        res = []
        stack = [(WHITE, root)]
        while stack:
            color, node = stack.pop()
            if node is None: continue
            if color == WHITE:
                stack.append((WHITE, node.right))
                stack.append((GRAY, node))
                stack.append((WHITE, node.left))
            else:
                res.append(node.val)
        return res
```

做法来自[题解](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/solution/yan-se-biao-ji-fa-yi-chong-tong-yong-qie-jian-ming/)：

* 使用颜色标记节点的状态，新节点为白色，已访问的节点为灰色
* 如果遇到的节点为白色，则将其标记为灰色，然后将其右子节点、自身、左子节点依次入栈
* 如果遇到的节点为灰色，则将节点的值输出
* 更换`color == WHITE`判断后的如栈顺序，可以完成前，中，后序遍历

# 39. 组合总和
[题目](https://leetcode-cn.com/problems/combination-sum/)

## 回溯法
```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res=[]
        candidates_copy=sorted(candidates)
        def find(ind,nums,tar):
            for i in range(ind,len(candidates)):
                n=candidates_copy[i]
                if tar==n:
                    res.append(nums+[n])
                if tar>n:
                    find(i,nums+[n],tar-n)
                if tar<n:
                    return
        find(0,[],target)
        return res
```
先对原数组排序，方便接下来的组合操作

从原数组的第一个元素开始添加，如果添加后仍为达到target，则继续对每个元素再进行添加

当第一个元素执行完上操作，对下一个元素执行；由于数组是排序过的，这种搜索方式可以避免重复的组合

# 114. 二叉树展开为链表

[题目](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/)

## 前序遍历储存位置

```python
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        l=list()
        def fun(node: TreeNode)->None:
            if not node:
                return
            else:
                l.append(node)
                fun(node.left)
                fun(node.right)
        fun(root)

        r=root
        for i in range(1,len(l)):
            r.left=None
            r.right=l[i]
            r=l[i]    
```

先用前序遍历存储每个树节点

遍历存储的列表，使每一个节点的右子节点是下一个元素

同时要注意把左子节点设置为`None`

## 寻找前驱法

```python
class Solution:
    def flatten(self, root: TreeNode) -> None:
        curr=root
        while curr:
            if curr.left:
                predecessor=nxt=curr.left
                while predecessor.right:
                    predecessor=predecessor.right
                predecessor.right = curr.right
                curr.left = None
                curr.right = nxt
            curr=curr.right
```

 对于每个节点：

* 不存在左子节点，则说明该节点和其右子节点是按照链表的顺序排列的，继续判断右子节点
* 存在左子节点，需要把右子节点挂到前驱节点（即该节点按前序遍历顺序时的上一个节点），同时使左子节点成为右子节点（注意最后使左子节点为空）