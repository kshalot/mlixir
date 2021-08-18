samples = 1_000
sample_length = 10

#TODO: Types
x64 = Nx.random_uniform({samples, sample_length})
y64 = Nx.random_uniform({samples})

x32 = Nx.random_uniform({samples, sample_length}, type: {:f, 32})
y32 = Nx.random_uniform({samples}, type: {:f, 32})

defmodule Benchmark.SGD do
  import Nx.Defn

  defn sgd(x, y) do
    Mlixir.SGD.fit(x, y)
    |> Mlixir.SGD.predict(x) #FIXME: Predicting the training set (shouldn't impact performance, but fix it later)
  end
end

benches = %{
  "elixir f64" => fn -> Benchmark.SGD.sgd(x64, y64) end,
  "elixir f32" => fn -> Benchmark.SGD.sgd(x32, y32) end
}

Benchee.run(
  benches,
  time: 10,
  memory_time: 2
)
