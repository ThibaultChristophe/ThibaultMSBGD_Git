// Set the Attributes of the Graph
var diameter = 650,
  format = d3.format(",d"),
  color1 = d3.scale.category20c();

var pack = d3.layout.pack()
  .size([diameter, diameter])
  .padding(1.5);

//Create SVG element
var svg1 = d3.select("#BubbleChart").append("svg")
  .attr("width", diameter)
  .attr("height", diameter)
  .attr("class", "bubble");


changebubble(8);


function changebubble(i) {
  d3.csv("count.csv", function(csvData) {
    var years = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014];
    pack.value(function(d) {
      return +d["count" + years[i]];
    });

    var data = {
      name: "city",
      children: csvData
    };

    var node = svg1.selectAll("g.node")
      .data(pack.nodes(data), function(d) {
        return d.city;
      });

    var nodeEnter = node.enter().append("g")
      .attr("class", "node")
      .attr("transform", function(d) {
        return "translate(" + d.x + "," + d.y + ")";
      });

    //Add the Circles
    var circles = nodeEnter.append("circle")
      .attr("r", function(d) {
        return d.r;
      })
      .style("fill", function(d) {
        return color1(d.city);
      });

    nodeEnter.append("title")
      .text(function(d) {
        return d.city + " : " + format(d.value);
      });

    //Add the Texts
    nodeEnter.append("text")
      .attr("dy", ".3em")
      .style("text-anchor", "middle")
      .text(function(d) {
        return d.city
      });

    nodeEnter.append("text")
      .attr("class", "value")
      .attr("dy", "1.3em")
      .style("text-anchor", "middle")
      .text(function(d) {
        return d.value
      });


    node.select("circle")
      .transition().duration(1000)
      .attr("r", function(d) {
        return d.r;
      })
      .style("fill", function(d) {
        return color1(d.city);
      });

    node.transition().duration(1000).attr("class", "node")
      .attr("transform", function(d) {
        return "translate(" + d.x + "," + d.y + ")";
      });

    node.select(".value")
      .text(function(d) {
        return format(d.value);
      });

    node.select("title")
      .text(function(d) {
        return d.city + " : " + format(d.value);
      });

    node.on('mouseover', highlightThisOne);
    node.on('mouseout', restoreAllColors);

    function highlightThisOne(d) {
      console.log(d);
      d3.selectAll("g.node").transition().duration(500)
        .attr("fill", function(d) {
          return color1(d.city);
        });
      d3.select(this).transition().duration(500)
        .attr('fill', "orange");
    }

    function restoreAllColors(d) {
      d3.selectAll("g.node").transition().duration(500)
        .attr("font-weight", "bold")
        .attr("fill", "black");
    }

    node.exit().remove();


  });
}

function updateBubble1() {
  changebubble(0);
}

function updateBubble2() {
  changebubble(1);
}

function updateBubble3() {
  changebubble(2);
}

function updateBubble4() {
  changebubble(3);
}

function updateBubble5() {
  changebubble(4);
}

function updateBubble6() {
  changebubble(5);
}

function updateBubble7() {
  changebubble(6);
}

function updateBubble8() {
  changebubble(7);
}

function updateBubble9() {
  changebubble(8);
}

d3.select("#yelp2006").on("click", updateBubble1);
d3.select("#yelp2007").on("click", updateBubble2);
d3.select("#yelp2008").on("click", updateBubble3);
d3.select("#yelp2009").on("click", updateBubble4);
d3.select("#yelp2010").on("click", updateBubble5);
d3.select("#yelp2011").on("click", updateBubble6);
d3.select("#yelp2012").on("click", updateBubble7);
d3.select("#yelp2013").on("click", updateBubble8);
d3.select("#yelp2014").on("click", updateBubble9);
