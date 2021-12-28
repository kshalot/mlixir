defmodule Mix.Tasks.Benchmarks.LinearRegression do
  use Mix.Task

  alias Mix.Tasks.Benchmarks.Dataset

  import Nx.Defn

  def run(_args) do
    # {train_images, train_labels} = Dataset.prepare_mnist()
    {train_images, train_labels} = {Nx.tensor([1]), Nx.tensor([4])}

    benches = %{
      "elixir" => fn -> elixir_fit(train_images, train_labels) end
      # "xla jit cpu" => fn -> host_fit(train_images, train_labels) end,
    }

    Benchee.run(
      benches,
      time: 10,
      memory_time: 2
    )
  end

  defn elixir_fit(train_images, train_labels) do
    Mlixir.LinearRegression.fit(train_images, train_labels)
  end

  # @defn_compiler EXLA
  # defn host_fit(train_images, train_labels) do
  #   elixir_fit(train_images, train_labels)
  # end
end
