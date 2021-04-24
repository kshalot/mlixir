defmodule Mlixir.DecisionTree do
  @moduledoc """
  Model representing a decision tree.
  """

  import Nx.Defn

  @behaviour Mlixir.Model

  alias Mlixir.Shared

  @doc """
  Train the decision tree.

  The result of this function are three tensors representing
  the Decision Tree:

  F - tensor of shape {attribute_count, node_count} encoding
      the mapping between nodes and input attributes.

  S - tensor of shape {node_count, class_count} encoding
      the parent relationship between nodes. It says whether
      the node is in the left or right subtree:
      1: left
      -1: right
      0: not a child

  C - tensor of shape {class_count} containg the values/classes
      in each terminal node.
  """
  @impl true
  defn fit(x, y) do
    transform(
      {x, y},
      fn {x, y} ->
        {_, n_classes} = Nx.shape(x)
      end
    )
  end

  defnp split(x, y, attribute_mask) do
    """
    The attribute_mask is used to denote which attributes
    were already used for splitting. 1 means that the attribute
    was used, 0 that it's available for splitting.
    """
  end

  defnp entropy(x) do
  end

  defnp information_gain(x) do
  end

  @doc """
  Predict using the trained model.
  """
  @impl true
  defn predict(model, x) do
  end
end
