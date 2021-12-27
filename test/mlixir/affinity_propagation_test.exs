defmodule Mlixir.AffinityPropagationTest do
  use ExUnit.Case

  @tiny_dataset Nx.tensor([[1, 2], [3, 4]])

  test "clusters the dataset" do
    assert Mlixir.AffinityPropagation.fit(@tiny_dataset) == 0
  end
end
