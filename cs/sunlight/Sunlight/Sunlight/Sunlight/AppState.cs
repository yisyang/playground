using Xamarin.Forms;
using System.Collections.Immutable;

// TODO: Not in use. Use Redux instead.
namespace Sunlight
{
    public class AppState
    {
        public string CurrentPage { get; set; } = string.Empty;
        public ImmutableArray<Tile> Tiles { get; set; } = ImmutableArray<Tile>.Empty;
    }

    public class Tile
    {
        public int x { get; set; }
        public int y { get; set; }
        public int elevation { get; set; }
        public Color color { get; set; }
    }

    public class NavigateAction
    {
        public string PageName { get; set; }
    }

    public class IncreaseElevationAction
    {
        
    }
}