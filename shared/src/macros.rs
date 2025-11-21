// pub use crate::potentials::Potential;

// #[macro_export]
// macro_rules! define_potential_enum {
//     (
//         bases: { $( $variant:ident ( $type:ty ) ),+ $(,)? },
//         wrappers: { $( $wrapper:ident ),* $(,)? }

//     ) => {
//         #[derive(Clone, Copy)]
//         pub enum PotentialEnum {
//             // Base variants
//             $(
//                 $variant($type),
//             )+

//             // Single wrappers: Wrapper<Base>
//             $(
//                 $(
//                     $wrapper$variant($wrapper<$type>),
//                 )+
//             )*

//             // Double wrappers: Wrapper2<Wrapper1<Base>>
//             $(
//                 $(
//                     $(
//                         $wrapper$variant$other($wrapper<$other<$type>>),
//                     )+
//                 )*
//             )*
//         }

//         impl $crate::potentials::Potential for PotentialEnum {
//             fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
//                 match self {
//                     // Base
//                     $(
//                         PotentialEnum::$variant(p) => p.force(t, x, y, z),
//                     )+

//                     // Single wrappers
//                     $(
//                         $(
//                             PotentialEnum::$wrapper$variant(p) => p.force(t, x, y, z),
//                         )+
//                     )*

//                     // Double wrappers
//                     $(
//                         $(
//                             $(
//                                 PotentialEnum::$wrapper$variant$other(p) => p.force(t, x, y, z),
//                             )+
//                         )*
//                     )*
//                 }
//             }
//         }
//     };
// }
