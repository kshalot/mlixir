defmodule Mlixir.DecisionTree do
  @moduledoc """
  Model representing a decision tree.
  """

  import Nx.Defn

  @behaviour Mlixir.Model

  @doc """
  Train the decision tree.

  The result of this function are four tensors representing
  the Decision Tree (allowing for GEMM traversal):

  F - tensor of shape {attribute_count, node_count} encoding
      the mapping between nodes and input attributes.

  S - tensor of shape {node_count, class_count} encoding
      the parent relationship between nodes. It says whether
      the node is in the left or right subtree:
      1: left
      -1: right
      0: not a child

  D - tensor of shape {class_count} encoding the number of
      "left" edges encountered during traversing up from the node
      to the root of the tree

  C - tensor of shape {class_count} containing the values/classes
      in each terminal node.
  """
  @impl true
  defn fit(x, y) do
    split(x, y)
   end

  defnp entropy(half1, half2) do
    # Calculate the entropy of a split.
  end

  defnp information_gain(x) do
    # Calculate information_gain of a split.
  end

  defnp split(x, y) do
    split_points = Nx.mean(x, axes: [1])
  end

  @doc """
  Predict using the trained model.

  The result for a given input 'x' is calculated with a simple formula,
  using the output tensors of the 'fit/2' method:

  ((x @ F @ S) == D) @ C
  """
  @impl true
  defn predict({f, s, d, c}, x) do
    x
    |> Nx.dot(f)
    |> Nx.dot(s)
    |> Nx.equal(d)
    |> Nx.dot(c)
  end
end
