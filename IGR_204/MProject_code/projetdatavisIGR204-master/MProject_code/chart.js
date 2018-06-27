var margin = {top: 20, right: 40, bottom: 10, left: 40},
        width = 500,
        height = 900 - margin.top - margin.bottom;

    var format = d3.format(".1f"), topTen, counts, Selection;

    var x = d3.scale.linear()
        .range([0,30]);

    var y = d3.scale.ordinal()
        .rangeRoundBands([0, height], .1);

    var color = d3.scale.ordinal().range(["#AC63CC","#7FCD50","#C95A3B","#59673E","#A4ACC9","#C85683","#CEB150","#8CCCA9","#504A76","#5F3333"]);

    var svg2 = d3.select("#chart").append("svg")
          .attr("width", width + margin.left + margin.right)
          .attr("height", height + margin.top + margin.bottom)
          .style("margin-left", -margin.left + "px")

    var menu = d3.select("#menu select")
        .on("change", change);

    d3.csv("count.csv", function(data) {
      Allcount = data;

      // Make array of the counts variables
      counts = d3.keys(Allcount[0]).filter(function(key) {
        return key != "city";
      });

      // Make variables numeric
      Allcount.forEach(function(elem) {
        counts.forEach(function(column) {
          elem[column] = +elem[column];
        });
      });

      // Load the counts type into drop-down
      menu.selectAll("option")
          .data(counts)
        .enter().append("option")
          .text(function(d) { return d; });

      // Set the intial value of drop-down when page loads
      menu.property("value", "count2014");

      // Call change function
      change();
    });

    function change() {
      Selection = menu.property("value"),
      topTen = Allcount.sort(function(a, b) { return b[Selection] - a[Selection]; }).slice(0, 100);
      render(topTen);
    }

    function render(data) {

      x.domain([0, data[0][Selection]]);
      y.domain(data.map(function(d) { return d.city; }));

      // Enter Selection
      svg2.selectAll(".circleGroup")
          .data(data, function(d) { return d.city; })
          .enter()
            .append("g")
              .attr("class", "circleGroup")
              .attr("transform", function(d) { return "translate(0," + (y(d.city) + height) + ")"; })
              .style("fill-opacity", .5)
              .each(function (d, i) {
                d3.select(this)
                  .append("circle")
                    .style("fill", function(d) { return color(d.city) })
                    .attr("cx", width / 2)
                    .attr("cy", y.rangeBand())
                    .attr("r", function(d) { return Math.log(d[Selection]); })

                d3.select(this)
                  .append("text")
                    .attr("class", "label")
                    .attr("x", (width / 2) + 60)
                    .attr("y", y.rangeBand() - 10)
                    .attr("dy", ".35em")
                    .attr("text-anchor", "start")
                    .text(function(d) { return d.city; })

                d3.select(this)
                  .append("text")
                    .attr("class", "value")
                    .attr("x", (width / 2) + 60)
                    .attr("y", y.rangeBand() + 10)
                    .attr("dy", ".35em")
                    .attr("text-anchor", "start");
                });

      // Update Selection
      svg2.selectAll(".circleGroup")
          .data(data, function(d) { return d.city; })
            .transition()
            .duration(2000)
            .attr("transform", function(d) { return "translate(0," + (d.y0 = y(d.city)) + ")"; })
            .style("fill-opacity", 1)
            .each(function (d, i) {
                d3.select(this)
                  .select("circle")
                  .attr("r", function(d) { return Math.log(d[Selection]); })

                d3.select(this)
                  .select(".value")
                  .text(function(d) { return format(d[Selection]); })
            });

      // Exit Selection
      svg2.selectAll(".circleGroup")
          .data(data, function(d) { return d.city; })
          .exit()
            .transition()
            .duration(2000)
            .attr("transform", function(d) { return "translate(0," + (d.y0 + height) + ")"; })
            .style("fill-opacity", 0)
            .remove();
    }
