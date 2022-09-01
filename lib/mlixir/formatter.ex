defmodule Formatter do
  def output(suite) do
    suite
    |> format
    |> IO.write()

    suite
  end

  defp format(suite) do
    Enum.map_join(suite.scenarios, "\n", fn scenario ->
      s = scenario.run_time_data.statistics
      "#{scenario.job_name},#{s.average}"
    end)
  end
end
