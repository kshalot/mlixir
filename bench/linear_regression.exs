samples = 5_000
sample_length = 10

#TODO: Types
x64 = Nx.random_uniform({samples, sample_length})
y64 = Nx.random_uniform({samples})

x32 = Nx.random_uniform({samples, sample_length}, type: {:f, 32})
y32 = Nx.random_uniform({samples}, type: {:f, 32})

defmodule Benchmark.LinearRegression do
  import Nx.Defn

  defn linear_regression(x, y) do
    Mlixir.LinearRegression.fit(x, y)
    |> Mlixir.LinearRegression.predict(x) #FIXME: Predicting the training set (shouldn't impact performance, but fix it later)
  end

  @defn_compiler EXLA
  defn host(x, y), do: linear_regression(x, y)
end

benches = %{
  "elixir f64" => fn -> Benchmark.LinearRegression.linear_regression(x64, y64) end,
  "elixir f32" => fn -> Benchmark.LinearRegression.linear_regression(x32, y32) end,
  "xla jit-cpu f64" => fn -> Benchmark.LinearRegression.host(x64, y64) end,
  "xla jit-cpu f32" => fn -> Benchmark.LinearRegression.host(x32, y32) end,
}

Benchee.run(
  benches,
  time: 10,
  memory_time: 2
)
