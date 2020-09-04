using System;
using System.Linq;
using Newtonsoft.Json;
using Xamarin.Forms;
using Xamarin.Forms.Xaml;

namespace Sunlight
{
    [XamlCompilation(XamlCompilationOptions.Compile)]
    public partial class HelloGrid
    {
        // TODO: Move everything to Redux
        private BoxView[,] _tileBoxViews2D = new BoxView[100, 100];
        private int[,] _tileElevation2D = new int[100, 100];
        private double[,] _tileSunlight2D = new double[100, 100];
        private const int XLimit = 6;
        private const int YLimit = 11;
        
        public HelloGrid()
        {
            InitializeComponent();

            int[] elevationSeed =
            {
                10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 5, 5, 5, 5, 5,
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10
            };
            
            Grid grid = new Grid
            {
                RowSpacing = 0,
                ColumnSpacing = 0
            };

            for (var yIndex = 0; yIndex < YLimit; yIndex++)
            {
                grid.RowDefinitions.Add(new RowDefinition());
            }

            for (var xIndex = 0; xIndex < XLimit; xIndex++)
            {
                grid.ColumnDefinitions.Add(new ColumnDefinition());                
            }

            // Prepare event listener
            var tgr = new TapGestureRecognizer();
            tgr.Tapped += OnGridTapped;

            // Create layout
            Console.WriteLine("Initializing Tiles.");
            for (var yIndex = 0; yIndex < YLimit; yIndex++)
            {
                for (var xIndex = 0; xIndex < XLimit; xIndex++)
                {
                    _tileElevation2D[xIndex, yIndex] = elevationSeed[(yIndex * 2 + xIndex * 3) % 40];

                    var boxView = new BoxView
                    {
                        Margin = new Thickness(20 - _tileElevation2D[xIndex, yIndex]),
                        Color = Color.Black,
                        WidthRequest = 10,
                        HeightRequest = 10
                    };
                    _tileBoxViews2D[xIndex, yIndex] = boxView;

                    // Register events
                    boxView.GestureRecognizers.Add(tgr);

                    grid.Children.Add(boxView, xIndex, yIndex);
                    grid.Children.Add(new Label
                    {
                        Text = $"{xIndex}, {yIndex}",
                        HorizontalOptions = LayoutOptions.Center,
                        VerticalOptions = LayoutOptions.Center
                    }, xIndex, yIndex);
                }                
            }
            
            // Compute sunlight
            Console.WriteLine("Initializing Sunlight.");
            for (var yIndex = 0; yIndex < YLimit; yIndex++)
            {
                for (var xIndex = 0; xIndex < XLimit; xIndex++)
                {
                    RecalculateSunlightAtPoint(xIndex, yIndex);
                }
            }

            Content = grid;
        }

        private void OnGridTapped(Object sender, EventArgs args)
        {
            var boxView = (BoxView) sender;
            var x = Grid.GetColumn(boxView);
            var y = Grid.GetRow(boxView);

            Console.WriteLine($"Tapped: ({x}, {y}).");

            _tileElevation2D[x, y] += 1;
            boxView.Margin = new Thickness(Math.Max(0, 20 - _tileElevation2D[x, y]));

            RecalculateSunlightAroundPoint(x, y);
        }

        private void RecalculateSunlightAroundPoint(int x, int y, int radius = 5)
        {
            var xMin = Math.Max(0, x - radius);
            var xMax = Math.Min(XLimit - 1, x + radius);
            var yMin = Math.Max(0, y - radius);
            var yMax = Math.Min(YLimit - 1, y + radius);
            for (var yIndex = yMin; yIndex <= yMax; yIndex++)
            {
                for (var xIndex = xMin; xIndex <= xMax; xIndex++)
                {
                    RecalculateSunlightAtPoint(xIndex, yIndex);
                }
            }
        }
        
        private void RecalculateSunlightAtPoint(int x, int y)
        {
            string sunlightRoutesJson = @"[
                [[0, 5], [0, 4], [0, 3], [0, 2], [0, 1]],
                [[1, 5], [1, 4], [1, 3], [0, 2], [0, 1]],
                [[2, 5], [2, 4], [1, 3], [1, 2], [0, 1]],
                [[3, 5], [2, 4], [2, 3], [1, 2], [1, 1]],
                [[4, 5], [3, 4], [2, 3], [2, 2], [1, 1]],
                [[5, 5], [4, 4], [3, 3], [2, 2], [1, 1]],
                [[5, 4], [4, 3], [3, 2], [2, 2], [1, 1]],
                [[5, 3], [4, 2], [3, 2], [2, 1], [1, 1]],
                [[5, 2], [4, 2], [3, 1], [2, 1], [1, 0]],
                [[5, 1], [4, 1], [3, 1], [2, 0], [1, 0]],
                [[5, 0], [4, 0], [3, 0], [2, 0], [1, 0]],
                [[5, -1], [4, -1], [3, 1], [2, 0], [1, 0]],
                [[5, -2], [4, -2], [3, -1], [2, -1], [1, 0]],
                [[5, -3], [4, -2], [3, -2], [2, -1], [1, -1]],
                [[5, -4], [4, -3], [3, -2], [2, -2], [1, -1]],
                [[5, -5], [4, -4], [3, -3], [2, -2], [1, -1]],
                [[4, -5], [3, -4], [2, -3], [2, -2], [1, -1]],
                [[3, -5], [2, -4], [2, -3], [1, -2], [1, -1]],
                [[2, -5], [2, -4], [1, -3], [1, -2], [0, -1]],
                [[1, -5], [1, -4], [1, -3], [0, -2], [0, -1]],
                [[0, -5], [0, -4], [0, -3], [0, -2], [0, -1]],
                [[-1, -5], [-1, -4], [-1, -3], [0, -2], [0, -1]],
                [[-2, -5], [-2, -4], [-1, -3], [-1, -2], [0, -1]],
                [[-3, -5], [-2, -4], [-2, -3], [-1, -2], [-1, -1]],
                [[-4, -5], [-3, -4], [-2, -3], [-2, -2], [-1, -1]],
                [[-5, -5], [-4, -4], [-3, -3], [-2, -2], [-1, -1]],
                [[-5, -4], [-4, -3], [-3, -2], [-2, -2], [-1, -1]],
                [[-5, -3], [-4, -2], [-3, -2], [-2, -1], [-1, -1]],
                [[-5, -2], [-4, -2], [-3, -1], [-2, -1], [-1, 0]],
                [[-5, -1], [-4, -1], [-3, 1], [-2, 0], [-1, 0]],
                [[-5, 0], [-4, 0], [-3, 0], [-2, 0], [-1, 0]],
                [[-5, 1], [-4, 1], [-3, 1], [-2, 0], [-1, 0]],
                [[-5, 2], [-4, 2], [-3, 1], [-2, 1], [-1, 0]],
                [[-5, 3], [-4, 2], [-3, 2], [-2, 1], [-1, 1]],
                [[-5, 4], [-4, 3], [-3, 2], [-2, 2], [-1, 1]],
                [[-5, 5], [-4, 4], [-3, 3], [-2, 2], [-1, 1]],
                [[-4, 5], [-3, 4], [-2, 3], [-2, 2], [-1, 1]],
                [[-3, 5], [-2, 4], [-2, 3], [-1, 2], [-1, 1]],
                [[-2, 5], [-2, 4], [-1, 3], [-1, 2], [0, 1]],
                [[-1, 5], [-1, 4], [-1, 3], [0, 2], [0, 1]]
            ]";
            var sunlightRoutes = JsonConvert.DeserializeObject<int[][][]>(sunlightRoutesJson);

            Console.WriteLine($"Evaluating Sunlight at ({x}, {y}).");
            var luminRoutes = new double[40];
            var elevationThis = _tileElevation2D[x, y];
            for (var i = 0; i < sunlightRoutes.Length; i++)
            {
                var route = sunlightRoutes[i];
                var deltaElevations = new double[5];
                var distance = 5;
                foreach (var pair in route)
                {
                    var xCompare = x + pair[0];
                    var yCompare = x + pair[1];
                    if (xCompare < 0 || xCompare >= XLimit || yCompare < 0 || yCompare >= YLimit)
                    {
                        // If out of bounds, assume no blocking.
                        deltaElevations[distance - 1] = -99;
                        distance--;
                        continue;
                    }
                    var elevationCompare = _tileElevation2D[xCompare, yCompare];
                    var deltaElevationOverDistance = (double)(elevationCompare - elevationThis) / distance;
                    deltaElevations[distance - 1] = deltaElevationOverDistance;
                    distance--;
                }
                // Lumin based on delta elevation: <=-1:1.2, 0:1, 1:0.5, 2:0.25, >=3:0
                var maxDe = deltaElevations.Max();
                if (maxDe < -1)
                {
                    luminRoutes[i] = 1.2;
                } else if (maxDe > 3)
                {
                    luminRoutes[i] = 0;
                }
                else
                {
                    luminRoutes[i] = Math.Max(0, 1 - 0.333 * maxDe - 0.1 * maxDe * maxDe + 0.033 * maxDe * maxDe * maxDe);
                }
            }
            
            var luminTile = luminRoutes.AsQueryable().Average();
            Console.WriteLine($"Lumin at ({x}, {y}) is {luminTile}.");
            _tileSunlight2D[x, y] = luminTile;
            _tileBoxViews2D[x, y].Color = Color.FromHsla(0, 0, luminTile / 1.2);
        }
    }
}